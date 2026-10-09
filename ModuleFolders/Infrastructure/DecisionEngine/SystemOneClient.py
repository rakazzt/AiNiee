"""HTTP client for a System One / JEV decision endpoint.

Jev is reachable through four provider surfaces. They share one request and response
shape (model + state + questions in, typed answers out) but differ in URL, and one of
them needs an account id and a restricted model name:

    systemone   POST {base}/v1/systemone                    TypeSafe direct, or OpenRouter
    decisions   POST {base}/alpha/decisions                 OpenRouter Decisions API
    cloudflare  POST .../accounts/{account_id}/ai/run/@cf/cloudflare/clef

Everything provider-specific is resolved into one URL here, so callers never build one.

Uses urllib rather than requests: this is a single POST, and the repo already does its
plain HTTP that way (HttpService, TaskConfig). urllib also honours HTTP(S)_PROXY.
"""

import json
from urllib import error as urlerror
from urllib import request as urlrequest

SYSTEMONE = "systemone"
DECISIONS = "decisions"
CLOUDFLARE = "cloudflare"
SHAPES = (SYSTEMONE, DECISIONS, CLOUDFLARE)

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_TIMEOUT = 30

# The Cloudflare model card restricts the selector to these two names.
CLOUDFLARE_MODELS = ("clef", "clef-flash")
CLOUDFLARE_API_ROOT = "https://api.cloudflare.com/client/v4/accounts"

_PATH_SUFFIXES = {
    SYSTEMONE: "/v1/systemone",
    DECISIONS: "/alpha/decisions",
}


class DecisionError(RuntimeError):
    """The decision endpoint could not be reached, or answered unusably.

    Carries what a log needs to still be useful months later: which kind of failure it
    was, the HTTP status when there was one, and the provider's own words. A caller that
    only prints str(error) still gets the old message.
    """

    def __init__(self, message, kind: str = "", status=None, detail: str = ""):
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.detail = detail


def build_endpoint(shape: str, base_url: str, account_id: str = "") -> str:
    """Resolve a provider shape into the one URL it posts to.

    A base that already carries the path is left alone, so pasting the full endpoint
    from the docs does not produce a doubled path.
    """
    base = (base_url or "").strip().rstrip("/")

    # A pasted Clef URL identifies its own provider, so a Cloudflare endpoint works even
    # when the interface was added as a plain custom platform.
    if base and "/ai/run/" in base:
        return base
    if shape not in SHAPES:
        raise DecisionError(
            "unknown provider shape {!r}; expected one of {}".format(shape, ", ".join(SHAPES)),
            kind="config",
        )

    if shape == CLOUDFLARE:
        account_id = (account_id or "").strip()
        if not account_id:
            raise DecisionError("the cloudflare shape needs an account id", kind="config")
        return "{}/{}/ai/run/@cf/cloudflare/clef".format(CLOUDFLARE_API_ROOT, account_id)

    suffix = _PATH_SUFFIXES[shape]
    if base.endswith(suffix) or base.endswith("/systemone") or base.endswith("/decisions"):
        return base
    if not base:
        base = DEFAULT_BASE_URL
    return base + suffix


class SystemOneClient:
    """One decision endpoint. Sends state + typed questions, returns typed answers."""

    def __init__(self, api_key: str, model: str, shape: str = SYSTEMONE,
                 base_url: str = DEFAULT_BASE_URL, account_id: str = "",
                 timeout: float = DEFAULT_TIMEOUT):
        self.api_key = (api_key or "").strip()
        self.model = (model or "").strip()
        self.shape = shape if shape in SHAPES else SYSTEMONE
        self.base_url = base_url or DEFAULT_BASE_URL
        self.account_id = account_id
        self.timeout = timeout or DEFAULT_TIMEOUT
        # The Cloudflare model card requires the selector in the body and accepts only
        # its own model names; catch that here rather than as an opaque 400.
        if self.shape == CLOUDFLARE and self.model and self.model not in CLOUDFLARE_MODELS:
            raise DecisionError(
                "the cloudflare shape accepts only {}; got {!r}".format(
                    " or ".join(CLOUDFLARE_MODELS), self.model
                ),
                kind="config",
            )

    @property
    def endpoint(self) -> str:
        return build_endpoint(self.shape, self.base_url, self.account_id)

    def evaluate(self, state, questions: dict, timeout: float = None) -> dict:
        """Ask every question about the state. Returns the raw response body.

        Raises DecisionError on any transport, status, or shape problem. Callers that
        must not fail the surrounding job are expected to catch it.
        """
        from ModuleFolders.Infrastructure.DecisionEngine import Questions
        Questions.validate(questions)
        if not self.api_key:
            raise DecisionError("no API key configured for the decision model", kind="config")
        if not self.model:
            raise DecisionError("no model configured for the decision model", kind="config")

        payload = {"model": self.model, "state": state, "questions": questions}
        body = _post_json(
            self.endpoint,
            {
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            payload,
            timeout or self.timeout,
        )
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise DecisionError("decision response has no answers map", kind="shape")
        return body


def _post_json(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urlrequest.Request(url, data=data, headers=headers, method="POST")
    try:
        with urlrequest.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urlerror.HTTPError as error:
        # The provider explains a rejected schema in the body; keep it for the log.
        detail = ""
        try:
            detail = error.read().decode("utf-8", "replace")[:400]
        except Exception:
            pass
        raise DecisionError(
            "decision endpoint returned HTTP {}: {}".format(error.code, detail),
            kind="http", status=error.code, detail=detail,
        )
    except (urlerror.URLError, OSError, ValueError) as error:
        # A timeout, a refused connection, a DNS failure: no HTTP status exists here, which
        # is exactly what tells a reader the request never got an answer.
        raise DecisionError("decision request failed: {}".format(error), kind="transport")
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise DecisionError(
            "decision endpoint returned invalid JSON: {}".format(error), kind="shape"
        )
    if not isinstance(body, dict):
        raise DecisionError(
            "decision endpoint returned {}, expected an object".format(type(body).__name__),
            kind="shape",
        )
    return body


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return None if value != value else value


def count_of(value):
    """A token count as a count: usage_of() normalises every number to float."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def usage_of(body: dict) -> dict:
    """Token and cost usage from a response, with the fields that may be absent zeroed.

    OpenRouter additionally returns usage.cost; the TypeSafe API does not.
    """
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    result = {
        "input_tokens": _number(usage.get("input_tokens")) or 0,
        "output_tokens": _number(usage.get("output_tokens")) or 0,
    }
    # Some surfaces report the price at the top level instead of inside usage.
    cost = _number(usage.get("cost"))
    if cost is None and isinstance(body, dict):
        cost = _number(body.get("cost"))
    if cost is not None:
        result["cost"] = cost
    return result
