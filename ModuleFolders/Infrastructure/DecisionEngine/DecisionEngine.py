"""The decision layer: ask a System One / JEV model narrow, typed questions.

Why this exists next to the translation requesters rather than inside them: Jev does
not generate text, so it does not fit the requester contract (messages in, prose out).
It answers typed questions, and code branches on the answer. That makes it the right
tool for the judgments AiNiee currently approximates with character-class rules.

Two deliberate policies:

  * Fail open. A decision gate must never fail a translation job because a classifier
    is unreachable. ask() returns None and the caller falls back to its local rule.
  * One request per chunk, many questions. Jev evaluates every question in a call
    independently, so batching is free accuracy and large savings on round trips.
"""

import json

from ModuleFolders.Infrastructure.DecisionEngine import Questions, SystemOneClient
from ModuleFolders.Infrastructure.DecisionEngine.SystemOneClient import DecisionError

# Questions are cheap; the state is what costs input tokens. Both ceilings are per
# request, so a long batch is split rather than silently truncated by the provider.
MAX_STATE_CHARS = 12000

# How many failed calls get described individually in the run log. Enough to show a
# pattern, bounded so an outage cannot turn the log into a wall of identical lines.
MAX_FAILURE_DETAIL = 20

_SHAPE_ALIASES = {
    "systemone": SystemOneClient.SYSTEMONE,
    "system one": SystemOneClient.SYSTEMONE,
    "typesafe": SystemOneClient.SYSTEMONE,
    "decisions": SystemOneClient.DECISIONS,
    "cloudflare": SystemOneClient.CLOUDFLARE,
    "clef": SystemOneClient.CLOUDFLARE,
}


def find_decision_platform(config) -> dict | None:
    """The configured decision model, or None if the user has not added one.

    Decision models live in the same platform store as the translation interfaces, so
    the API key, endpoint and test button are the ones that already exist. They carry
    group "decision" and are kept out of the translation platform pickers.
    """
    if not isinstance(config, dict):
        return None
    platforms = config.get("platforms")
    if not isinstance(platforms, dict):
        return None
    for platform in platforms.values():
        if isinstance(platform, dict) and platform.get("group") == "decision":
            return platform
    return None


def resolve_shape(api_format: str) -> str:
    """Map a declared api_format onto a provider shape, defaulting to systemone."""
    return _SHAPE_ALIASES.get(str(api_format or "").strip().lower(), SystemOneClient.SYSTEMONE)


class DecisionEngine:
    """A configured decision endpoint plus the counters needed to account for it."""

    def __init__(self, client: SystemOneClient.SystemOneClient):
        self.client = client
        self.calls = 0
        self.failures = 0
        self.questions_asked = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.last_error = ""
        # Per-failure detail, bounded. A count alone cannot be acted on months later;
        # the HTTP status and the provider's own words can.
        self.failure_detail: list = []

    @classmethod
    def from_platform(cls, platform: dict) -> "DecisionEngine":
        if not isinstance(platform, dict):
            raise DecisionError("decision platform config must be a map")
        try:
            timeout = float(platform.get("timeout", SystemOneClient.DEFAULT_TIMEOUT))
        except (TypeError, ValueError):
            timeout = SystemOneClient.DEFAULT_TIMEOUT
        client = SystemOneClient.SystemOneClient(
            api_key=platform.get("api_key", ""),
            model=platform.get("model", ""),
            shape=resolve_shape(platform.get("api_format")),
            base_url=platform.get("api_url") or SystemOneClient.DEFAULT_BASE_URL,
            account_id=platform.get("account_id", ""),
            timeout=timeout,
        )
        return cls(client)

    @classmethod
    def from_config(cls, config) -> "DecisionEngine | None":
        """Build from config, or None when no decision model is configured."""
        platform = find_decision_platform(config)
        if platform is None:
            return None
        try:
            return cls.from_platform(platform)
        except DecisionError:
            return None

    # --- asking ---------------------------------------------------------------

    def ask(self, state, questions: dict, purpose: str = "") -> dict | None:
        """Ask every question about one state. Returns answers, or None on failure.

        Never raises for a provider problem: that is the fail-open policy. A caller that
        needs to distinguish "answered no" from "could not ask" tests for None.
        """
        self.calls += 1
        self.questions_asked += len(questions) if isinstance(questions, dict) else 0
        try:
            body = self.client.evaluate(state, questions)
        except DecisionError as error:
            self._record_failure(purpose, error)
            return None
        self._record_usage(body)
        return body.get("answers")

    def decide_batch(self, items, build, purpose: str = "", extra_state: dict = None,
                     max_state_chars: int = MAX_STATE_CHARS) -> list:
        """Ask per-item questions about many items, batched into as few calls as allowed.

        build(index, item) -> (state_fragment, questions), where index is the item's
        position in items. Each fragment is placed at state["items"]["<index>"], so a
        question can name the value it is about with a path such as items.3.translated,
        and that path stays correct however the batch is split.

        Returns a list aligned with items. Each entry is the answers map for the request
        that carried it, or {} when that request failed, so a reader yields None.
        """
        items = list(items)
        results: list = [{} for _ in items]
        if not items:
            return results

        for chunk in self._plan_chunks(items, build, max_state_chars):
            state = dict(extra_state) if extra_state else {}
            state["items"] = {str(index): fragment for index, fragment, _ in chunk}
            questions = {}
            for _, _, item_questions in chunk:
                questions.update(item_questions)
            answers = self.ask(state, questions, purpose)
            if answers is None:
                continue  # fail open: these entries stay {} and read as unknown
            for index, _, _ in chunk:
                results[index] = answers
        return results

    def _plan_chunks(self, items, build, max_state_chars: int) -> list:
        """Group items so no single request breaks a provider ceiling."""
        chunks: list = []
        current: list = []
        question_count = 0
        char_count = 0

        for index, item in enumerate(items):
            fragment, questions = build(index, item)
            Questions.validate(questions)
            weight = self._weight(fragment, questions)
            if current and (
                question_count + len(questions) > Questions.MAX_QUESTIONS
                or char_count + weight > max_state_chars
            ):
                chunks.append(current)
                current, question_count, char_count = [], 0, 0
            # An oversized item still goes alone: it cannot be split further, and the
            # provider will truncate the state rather than reject the request.
            current.append((index, fragment, questions))
            question_count += len(questions)
            char_count += weight
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _weight(fragment, questions) -> int:
        return len(json.dumps([fragment, questions], ensure_ascii=False))

    # --- accounting -----------------------------------------------------------

    def _record_usage(self, body: dict) -> None:
        usage = SystemOneClient.usage_of(body)
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        self.cost += usage.get("cost", 0.0)

    def _record_failure(self, purpose: str, error: Exception) -> None:
        """Count a failure and keep enough about it to diagnose it later."""
        self.failures += 1
        self.last_error = str(error)
        if purpose:
            self.last_purpose = purpose
        if len(self.failure_detail) < MAX_FAILURE_DETAIL:
            self.failure_detail.append({
                "purpose": purpose,
                "kind": getattr(error, "kind", ""),
                "status": getattr(error, "status", None),
                "detail": getattr(error, "detail", ""),
                "error": str(error),
            })

    def summary(self) -> dict:
        """Counters for a log line. Cheap enough to print at the end of a job."""
        return {
            "calls": self.calls,
            "failures": self.failures,
            "questions": self.questions_asked,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": round(self.cost, 6),
            "last_error": self.last_error,
            "model": self.model,
            "endpoint": self.endpoint,
            "shape": self.shape,
            "failure_detail": [dict(record) for record in self.failure_detail],
        }

    # --- what the run log says ------------------------------------------------

    @property
    def model(self) -> str:
        return getattr(self.client, "model", "")

    @property
    def endpoint(self) -> str:
        return getattr(self.client, "endpoint", "")

    @property
    def shape(self) -> str:
        return getattr(self.client, "shape", "")

    def describe(self) -> str:
        """Which decision model this is, and what it did — for the run log.

        Printing this whether or not anything failed is the point: a silent fail-open and
        a decision layer that was never used otherwise read exactly the same.
        """
        return (
            "决策模型：{model}（{shape}）\n"
            "接口地址：{endpoint}\n"
            "调用 {calls} 次，提问 {questions} 个，失败 {failures} 次，"
            "输入 {input_tokens} Tokens，输出 {output_tokens} Tokens，花费 {cost} USD"
        ).format(
            model=self.model or "(未填写模型)",
            shape=self.shape or "unknown",
            endpoint=self.endpoint or "(未解析)",
            calls=self.calls,
            questions=self.questions_asked,
            failures=self.failures,
            input_tokens=SystemOneClient.count_of(self.input_tokens),
            output_tokens=SystemOneClient.count_of(self.output_tokens),
            # Fixed notation: a log line reading 8.1e-05 is not a number anyone checks.
            cost="{:.8f}".format(self.cost),
        )

    def failure_lines(self) -> list:
        """One line per recorded failure, with its kind, HTTP status and provider words."""
        lines = []
        for record in self.failure_detail:
            where = record["purpose"] or "未标注用途"
            kind = record["kind"] or "unknown"
            status = record["status"]
            label = kind if status is None else "{}/HTTP {}".format(kind, status)
            lines.append("决策模型调用失败（{}，{}）：{}".format(where, label, record["error"]))
        hidden = self.failures - len(self.failure_detail)
        if hidden > 0:
            lines.append("（另有 {} 次失败未逐条记录，只计入总数）".format(hidden))
        return lines
