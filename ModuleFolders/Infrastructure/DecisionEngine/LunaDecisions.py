"""GPT-6 Luna Decisions: typed questions over OpenRouter's Decisions router.

The model answers typed questions about evidence you send it and never writes prose, so it
does not fit the translation requesters (messages in, text out). It belongs with the other
decision models, and this module is its typed front door.

OpenRouter serves it on /api/alpha/decisions, and that route's wire format is not the one
the model's own OpenAI guide documents:

    OpenAI /v1/decisions                     OpenRouter /api/alpha/decisions
    input: text | messages[input_text|..]    state: string | object | array
    questions: [ {type: predicate|..} ]      questions: { name: {type: noul|..} }
    answers: [ .. ] in question order        answers: { name: .. }

The OpenRouter route requires model + state + questions, and its multimodal guide rejects
the OpenAI input part shapes outright, so an OpenAI-shaped body is a 400 there. This module
speaks that wire and exposes the typed question model from the OpenAI guide on top of it:
predicate / choice / score, each optionally named, read back in the order they were asked.

Two policies match the rest of DecisionEngine:

  * An answer that cannot be read is unavailable, never "no". A refusal is reported as a
    refusal, never as a value.
  * One attempt, bounded. There is no retry and no Idempotency-Key: a request that timed
    out may already be billed, so asking twice spends real money twice. ask() raises and
    leaves the decision to the caller.

Every input problem raises Questions.QuestionError, because each one is a request the API
would reject rather than answer.
"""

import json
import os
import sys

from ModuleFolders.Infrastructure.DecisionEngine import Questions, SystemOneClient
from ModuleFolders.Infrastructure.DecisionEngine.DecisionEngine import MAX_STATE_CHARS
from ModuleFolders.Infrastructure.DecisionEngine.SystemOneClient import DecisionError

PREDICATE = "predicate"
CHOICE = "choice"
SCORE = "score"
QUESTION_TYPES = (PREDICATE, CHOICE, SCORE)

REFUSAL = "refusal"
UNAVAILABLE = "unavailable"

DEFAULT_MODEL = "openai/gpt-6-luna-decisions"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
DEFAULT_TIMEOUT = 60

# Documented for this model on the OpenRouter multimodal guide: at most 128 images per
# request, each an inline base64 data URL. Remote http(s) URLs are never fetched and a
# top-level "images" field is a 400, so both are refused here with a readable message.
MAX_IMAGES = 128
IMAGE_DATA_PREFIXES = (
    "data:image/png;base64,",
    "data:image/jpeg;base64,",
    "data:image/webp;base64,",
)

_QUESTION_FIELDS = frozenset({"type", "name", "instructions", "choices", "levels"})


# --- Writing questions ---------------------------------------------------------

def _label(value, what):
    if not isinstance(value, str) or not value.strip():
        raise Questions.QuestionError("{} must be a non-empty string".format(what))
    return value


def _instructions(value):
    if isinstance(value, (dict, list)):
        return value
    return _label(value, "instructions")


def _named(question, name):
    if name is not None:
        question["name"] = _label(name, "a question name")
    return question


def predicate(instructions, name=None):
    """Does the statement hold? The answer is the probability that it is true."""
    return _named({"type": PREDICATE, "instructions": _instructions(instructions)}, name)


def choice(instructions, choices, name=None):
    """Which of these is it? The answer names one of the supplied values."""
    if not isinstance(choices, (list, tuple)) or not choices:
        raise Questions.QuestionError("choices must be a non-empty list of {value, description}")
    cleaned = []
    seen = set()
    for option in choices:
        if not isinstance(option, dict):
            raise Questions.QuestionError("each choice must be a {value, description} object")
        unknown = set(option) - {"value", "description"}
        if unknown:
            raise Questions.QuestionError(
                "a choice has unknown field(s): {}".format(", ".join(sorted(unknown)))
            )
        value = _label(option.get("value"), "a choice value")
        if value in seen:
            raise Questions.QuestionError("duplicate choice value: {!r}".format(value))
        seen.add(value)
        description = option.get("description")
        cleaned.append({
            "value": value,
            "description": None if description is None else _label(description, "a choice description"),
        })
    return _named(
        {"type": CHOICE, "instructions": _instructions(instructions), "choices": cleaned}, name
    )


def score(instructions, levels, name=None):
    """How much? The answer is a position on the ordered levels, lowest first."""
    if not isinstance(levels, (list, tuple)):
        raise Questions.QuestionError("levels must be an ordered list of {label, description}")
    if not (Questions.MIN_SCORE_LEVELS <= len(levels) <= Questions.MAX_SCORE_LEVELS):
        raise Questions.QuestionError(
            "score needs {}..{} levels, got {}".format(
                Questions.MIN_SCORE_LEVELS, Questions.MAX_SCORE_LEVELS, len(levels)
            )
        )
    cleaned = []
    for level in levels:
        if not isinstance(level, dict):
            raise Questions.QuestionError("each level must be a {label, description} object")
        unknown = set(level) - {"label", "description"}
        if unknown:
            raise Questions.QuestionError(
                "a level has unknown field(s): {}".format(", ".join(sorted(unknown)))
            )
        description = level.get("description")
        cleaned.append({
            "label": _label(level.get("label"), "a level label"),
            "description": None if description is None else _label(description, "a level description"),
        })
    return _named(
        {"type": SCORE, "instructions": _instructions(instructions), "levels": cleaned}, name
    )


# --- Encoding onto the OpenRouter wire -----------------------------------------

def _level_text(level):
    """One score level as the wire carries it, which is a description and nothing else.

    The OpenAI shape has a label and a description per level; OpenRouter's score criteria
    is a list of strings. Folding the label in keeps it visible to the model and makes the
    returned legend self-describing, which is what describe() prints.
    """
    label, description = level.get("label"), level.get("description")
    if description:
        return "{}: {}".format(label, description)
    return label


def _to_wire_question(question):
    """Map one typed question onto the question body OpenRouter accepts."""
    kind = question["type"]
    instructions = question.get("instructions")
    if kind == PREDICATE:
        return Questions.noul(instructions)
    if kind == CHOICE:
        criteria = {}
        for option in question.get("choices") or []:
            criteria[option["value"]] = option.get("description")
        return Questions.choice(instructions, criteria)
    return Questions.score(instructions, [_level_text(level) for level in question.get("levels") or []])


def _encode(questions):
    """Encode typed questions into the wire map. Returns (keys, questions_map).

    keys[i] is the wire key for questions[i], so answers can be read back in the order the
    questions were asked however the caller named them. An unnamed question gets a
    positional key, and its answer is reported without a name, as the API does.
    """
    if not isinstance(questions, (list, tuple)) or not questions:
        raise Questions.QuestionError("questions must be a non-empty list")
    if len(questions) > Questions.MAX_QUESTIONS:
        raise Questions.QuestionError(
            "at most {} questions per request, got {}".format(Questions.MAX_QUESTIONS, len(questions))
        )

    keys = []
    encoded = {}
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise Questions.QuestionError("question {} must be a map".format(index))
        unknown = set(question) - _QUESTION_FIELDS
        if unknown:
            raise Questions.QuestionError(
                "question {} has unknown field(s): {}".format(index, ", ".join(sorted(unknown)))
            )
        kind = question.get("type")
        if kind not in QUESTION_TYPES:
            raise Questions.QuestionError(
                "question {} has unknown type {!r}; expected {}".format(
                    index, kind, ", ".join(QUESTION_TYPES)
                )
            )
        name = question.get("name")
        if name is None:
            key = "q{}".format(index)
        else:
            key = _label(name, "a question name")
            if not Questions.QUESTION_ID_RE.match(key):
                raise Questions.QuestionError(
                    "question name {!r} must match {}".format(key, Questions.QUESTION_ID_RE.pattern)
                )
        if key in encoded:
            raise Questions.QuestionError(
                "question key {!r} is used twice; give every question a distinct name".format(key)
            )
        keys.append(key)
        encoded[key] = _to_wire_question(question)
    return keys, encoded


# --- Encoding the evidence -----------------------------------------------------

def _part(part):
    kind = part.get("type")
    if kind in ("input_text", "text"):
        return _label(part.get("text"), "an input_text part needs text")
    if kind == "input_image":
        url = part.get("image_url")
        if not isinstance(url, str) or not url:
            raise Questions.QuestionError("an input_image part needs an image_url data URL")
        if not url.startswith(IMAGE_DATA_PREFIXES):
            raise Questions.QuestionError(
                "images must be inline base64 data URLs (image/png, image/jpeg or image/webp); "
                "remote URLs and file ids are not fetched by this endpoint"
            )
        return {"type": "image_url", "image_url": {"url": url}}
    raise Questions.QuestionError(
        "unsupported input part type {!r}; expected input_text or input_image".format(kind)
    )


def _flatten(item, out):
    if isinstance(item, str):
        out.append(item)
        return
    if not isinstance(item, dict):
        raise Questions.QuestionError("an input item must be text or a part object")
    if "role" in item:
        content = item.get("content")
        if isinstance(content, str):
            out.append(content)
            return
        if not isinstance(content, (list, tuple)):
            raise Questions.QuestionError("a message must carry a string or a list of parts")
        for part in content:
            _flatten(part, out)
        return
    out.append(_part(item))


def encode_state(content):
    """Encode the evidence into the state OpenRouter evaluates.

    Accepts a plain string, an object, a list of parts, or the OpenAI message list
    [{role, content: [{type: input_text|input_image, ..}]}]. Text becomes a plain string
    item and each image becomes an image_url part, which is the only shape this route
    reads: a text part object is not read, a top-level images field is a 400, and a remote
    URL is never fetched.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content
    if not isinstance(content, (list, tuple)):
        raise Questions.QuestionError("input must be a string, an object, or a list of parts")

    parts = []
    for item in content:
        _flatten(item, parts)
    if not parts:
        raise Questions.QuestionError("input must not be empty")
    images = sum(1 for part in parts if isinstance(part, dict))
    if images > MAX_IMAGES:
        raise Questions.QuestionError(
            "at most {} images per request, got {}".format(MAX_IMAGES, images)
        )
    return parts


def review_state(diff, test_summary, max_chars=MAX_STATE_CHARS):
    """The evidence for a review question: the diff and a test summary, not the repository.

    Truncated at max_chars, because the state is what the request is billed on and a whole
    repository would neither fit nor help.
    """
    text = "Diff:\n{}\n\nTest summary:\n{}".format(diff or "(none)", test_summary or "(none)")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[truncated at {} characters]".format(max_chars)


# --- Reading the answers -------------------------------------------------------

class Answer:
    """One answer, read at the position of the question that asked for it."""

    __slots__ = ("type", "name", "value", "confidence", "distribution", "legend")

    def __init__(self, answer_type, name=None, value=None, confidence=None,
                 distribution=None, legend=None):
        self.type = answer_type
        self.name = name
        self.value = value
        self.confidence = confidence
        self.distribution = distribution
        self.legend = legend

    @property
    def is_refusal(self):
        return self.type == REFUSAL

    @property
    def is_available(self):
        return self.type in QUESTION_TYPES

    def describe(self):
        label = self.name or "(unnamed)"
        if self.type == PREDICATE:
            return "{}: probability {:.4f}".format(label, self.value)
        if self.type == CHOICE:
            return "{}: {}{}".format(label, self.value, _confidence_suffix(self.confidence))
        if self.type == SCORE:
            return "{}: score {:.3f}{}".format(label, self.value, _confidence_suffix(self.confidence))
        if self.type == REFUSAL:
            return "{}: refused by the model".format(label)
        return "{}: no answer".format(label)

    def __repr__(self):
        return "Answer(type={!r}, {})".format(self.type, self.describe())


def _confidence_suffix(confidence):
    return "" if confidence is None else " (confidence {:.2f})".format(confidence)


class Decision:
    """The answers in question order, plus what the request was reported to cost."""

    def __init__(self, answers, usage=None, model="", id="", provider="", raw=None):
        self.answers = list(answers)
        self.usage = usage or {}
        self.model = model
        self.id = id
        self.provider = provider
        self.raw = raw

    def answer_at(self, index):
        """The answer at a question's position, or None when there is no such position."""
        if isinstance(index, bool) or not isinstance(index, int):
            return None
        if index < 0 or index >= len(self.answers):
            return None
        return self.answers[index]

    def describe(self):
        return [answer.describe() for answer in self.answers]

    def usage_line(self):
        text = "{} input tokens, {} output tokens".format(
            SystemOneClient.count_of(self.usage.get("input_tokens", 0)),
            SystemOneClient.count_of(self.usage.get("output_tokens", 0)),
        )
        cost = self.usage.get("cost")
        if cost is not None:
            text += ", cost {:.8f} USD".format(cost)
        return text


def _read(question, key, answers):
    """Read one answer. Anything unreadable is unavailable, never a value."""
    name = question.get("name")
    raw = answers.get(key)
    if not isinstance(raw, dict):
        return Answer(UNAVAILABLE, name)
    if raw.get("type") == REFUSAL:
        return Answer(REFUSAL, name)

    kind = question.get("type")
    if kind == PREDICATE:
        value = Questions.noul_probability(answers, key)
        return Answer(UNAVAILABLE, name) if value is None else Answer(PREDICATE, name, value=value)
    if kind == CHOICE:
        value = Questions.choice_label(answers, key)
        if value is None:
            return Answer(UNAVAILABLE, name)
        return Answer(CHOICE, name, value=value,
                      confidence=Questions.confidence(answers, key),
                      distribution=Questions.choice_probabilities(answers, key))
    value = Questions.score_value(answers, key)
    if value is None:
        return Answer(UNAVAILABLE, name)
    legend = raw.get("legend")
    return Answer(SCORE, name, value=value,
                  confidence=Questions.confidence(answers, key),
                  distribution=Questions.score_probabilities(answers, key),
                  legend=legend if isinstance(legend, dict) else None)


# --- The endpoint --------------------------------------------------------------

class LunaDecisions:
    """GPT-6 Luna Decisions: typed questions in, typed answers out, one attempt."""

    def __init__(self, api_key="", model=DEFAULT_MODEL, base_url=OPENROUTER_BASE_URL,
                 timeout=DEFAULT_TIMEOUT, client=None):
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.client = client if client is not None else SystemOneClient.SystemOneClient(
            api_key=api_key,
            model=model,
            shape=SystemOneClient.DECISIONS,
            base_url=base_url or OPENROUTER_BASE_URL,
            timeout=self.timeout,
        )

    def ask(self, content, questions, timeout=None):
        """Ask every question about content, in one request.

        Raises Questions.QuestionError when the request could never be accepted, and
        DecisionError when the endpoint could not be reached or answered unusably. There
        is no retry: a decision call that timed out may already be billed.
        """
        keys, encoded = _encode(questions)
        state = encode_state(content)
        body = self.client.evaluate(state, encoded, timeout=timeout or self.timeout)
        answers = body.get("answers")
        if not isinstance(answers, dict):
            answers = {}
        return Decision(
            [_read(question, key, answers) for question, key in zip(questions, keys)],
            usage=SystemOneClient.usage_of(body),
            model=body.get("model", ""),
            id=body.get("id", ""),
            provider=body.get("provider", ""),
            raw=body,
        )


def main(argv=None):
    """Ask the questions in a request file, then print the answers and the usage.

        python -m ModuleFolders.Infrastructure.DecisionEngine.LunaDecisions request.json

    The file is {"input": ..., "questions": [...]} with optional model and timeout. The
    key comes from OPENROUTER_API_KEY. This makes one real, billed request.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m ModuleFolders.Infrastructure.DecisionEngine.LunaDecisions request.json")
        return 2
    with open(argv[0], "r", encoding="utf-8") as handle:
        request = json.load(handle)
    client = LunaDecisions(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        model=request.get("model", DEFAULT_MODEL),
        timeout=float(request.get("timeout", DEFAULT_TIMEOUT)),
    )
    decision = client.ask(request.get("input", ""), request.get("questions", []))
    for line in decision.describe():
        print(line)
    print("usage: " + decision.usage_line())
    return 0


if __name__ == "__main__":
    sys.exit(main())
