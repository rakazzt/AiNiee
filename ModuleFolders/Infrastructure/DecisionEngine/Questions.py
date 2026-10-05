"""Typed questions for a System One / JEV decision model.

Jev is not a chat model. It never returns prose: you send a state and a map of
typed questions, and it returns one typed answer per question, with a probability
distribution. Code branches on that value directly, so there is nothing to parse.

This module owns the three primitives and the documented schema limits, so callers
describe a decision instead of writing a prompt.

The readers at the bottom return None for anything they cannot understand. That is
deliberate: a missing answer must never be read as "no", because a classifier
outage would then look like "every translation is wrong" and trigger a retry storm.
"""

import re

# Documented limits: docs.typesafe.ai/api (Choice <= 255 options, Score 2..10
# levels) and the Cloudflare clef model card (1..64 questions per request).
MAX_QUESTIONS = 64
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

# Cloudflare documents this id shape; the TypeSafe API adds no constraint of its
# own, so the narrower rule is the one enforced here.
QUESTION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")

NOUL = "noul"
CHOICE = "choice"
SCORE = "score"

_ALLOWED_FIELDS = frozenset({"type", "instructions", "criteria"})


class QuestionError(ValueError):
    """A question would be rejected by the API, or silently change meaning."""


def _instructions(value):
    if isinstance(value, str):
        if not value.strip():
            raise QuestionError("instructions must not be blank")
        return value
    # The API also accepts an object or array, which is how a question points at
    # data it needs by name. Anything else would be sent as a useless literal.
    if isinstance(value, (dict, list)):
        return value
    raise QuestionError("instructions must be a string, object or array")


def noul(instructions, true_meaning=None, false_meaning=None) -> dict:
    """A yes/no question. Returns the probability that the answer is yes."""
    question = {"type": NOUL, "instructions": _instructions(instructions)}
    if true_meaning is not None or false_meaning is not None:
        criteria = {}
        if true_meaning is not None:
            criteria["true"] = _instructions(true_meaning)
        if false_meaning is not None:
            criteria["false"] = _instructions(false_meaning)
        question["criteria"] = criteria
    return question


def choice(instructions, options: dict) -> dict:
    """Pick one option. Returns the chosen option plus every probability."""
    if not isinstance(options, dict) or not options:
        raise QuestionError("choice criteria must be a non-empty map of option to description")
    if len(options) > MAX_CHOICE_OPTIONS:
        raise QuestionError(
            "choice allows at most {} options, got {}".format(MAX_CHOICE_OPTIONS, len(options))
        )
    criteria = {}
    for option, description in options.items():
        if not isinstance(option, str) or not option.strip():
            raise QuestionError("choice option keys must be non-empty strings")
        # None is legal and means this option needs no extra detail.
        criteria[option] = None if description is None else _instructions(description)
    return {"type": CHOICE, "instructions": _instructions(instructions), "criteria": criteria}


def score(instructions, levels) -> dict:
    """Rate the state against ordered levels, lowest first. Returns a position."""
    if not isinstance(levels, (list, tuple)):
        raise QuestionError("score criteria must be an ordered list of levels")
    if not (MIN_SCORE_LEVELS <= len(levels) <= MAX_SCORE_LEVELS):
        raise QuestionError(
            "score needs {}..{} levels, got {}".format(
                MIN_SCORE_LEVELS, MAX_SCORE_LEVELS, len(levels)
            )
        )
    return {
        "type": SCORE,
        "instructions": _instructions(instructions),
        "criteria": [_instructions(level) for level in levels],
    }


def validate(questions: dict) -> dict:
    """Check a whole question map before it is sent, and return it unchanged.

    Strict on purpose: an unknown field (a typo such as "critera") would otherwise
    be dropped by the API and quietly change what was asked.
    """
    if not isinstance(questions, dict) or not questions:
        raise QuestionError("questions must be a non-empty map")
    if len(questions) > MAX_QUESTIONS:
        raise QuestionError(
            "at most {} questions per request, got {}".format(MAX_QUESTIONS, len(questions))
        )
    for qid, question in questions.items():
        if not isinstance(qid, str) or not QUESTION_ID_RE.match(qid):
            raise QuestionError("invalid question id: {!r}".format(qid))
        if not isinstance(question, dict):
            raise QuestionError("question {!r} must be a map".format(qid))
        qtype = question.get("type")
        if qtype not in (NOUL, CHOICE, SCORE):
            raise QuestionError("question {!r} has unknown type {!r}".format(qid, qtype))
        unknown = set(question) - _ALLOWED_FIELDS
        if unknown:
            raise QuestionError(
                "question {!r} has unknown field(s): {}".format(qid, ", ".join(sorted(unknown)))
            )
        if "instructions" not in question:
            raise QuestionError("question {!r} is missing instructions".format(qid))
        if qtype in (CHOICE, SCORE) and "criteria" not in question:
            raise QuestionError("{} question {!r} needs criteria".format(qtype, qid))
    return questions


# --- Reading answers -----------------------------------------------------------
# Every reader returns None when the answer is absent or is not the type that was
# asked for. Callers must treat None as unknown and fall back, never as a "no".

def _answer(answers, qid):
    if not isinstance(answers, dict):
        return None
    answer = answers.get(qid)
    return answer if isinstance(answer, dict) else None


def _probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if value != value:  # NaN
        return None
    return min(1.0, max(0.0, value))


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return None if value != value else value


def noul_probability(answers, qid):
    """Probability in [0, 1] that the answer is yes, or None if unavailable."""
    answer = _answer(answers, qid)
    if answer is None or answer.get("type") != NOUL:
        return None
    return _probability(answer.get(NOUL))


def choice_label(answers, qid):
    """The highest-probability option, or None if unavailable."""
    answer = _answer(answers, qid)
    if answer is None or answer.get("type") != CHOICE:
        return None
    label = answer.get(CHOICE)
    if not isinstance(label, str) or not label:
        return None
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and label not in probabilities:
        # The label is not one of the options we offered: not a usable answer.
        return None
    return label


def choice_probabilities(answers, qid):
    """Every option mapped to its probability, or None if unavailable."""
    answer = _answer(answers, qid)
    if answer is None or answer.get("type") != CHOICE:
        return None
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict):
        return None
    cleaned = {}
    for option, value in probabilities.items():
        probability = _probability(value)
        if probability is not None:
            cleaned[option] = probability
    return cleaned or None


def score_value(answers, qid):
    """Probability-weighted position on the scale (0 = first level), or None."""
    answer = _answer(answers, qid)
    if answer is None or answer.get("type") != SCORE:
        return None
    return _number(answer.get(SCORE))


def confidence(answers, qid):
    """How certain the model is about a choice or score answer, or None."""
    answer = _answer(answers, qid)
    if answer is None:
        return None
    return _probability(answer.get("confidence"))
