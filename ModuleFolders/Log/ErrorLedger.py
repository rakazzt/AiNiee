"""Fold repeated identical errors so one outage cannot bury the log.

A provider outage repeats one message hundreds of times. Logging every repeat with its
traceback buries the few lines that matter: a user saw 184 identical 503s and could not
tell a slow run from a dead one. This module keeps the counts instead - the caller logs
the first occurrence in full, repeats are counted silently, and every REPEAT_EVERY-th
surfaces carrying the running count. Nothing is discarded; summary() reports the total.

Kept free of rich and logging imports on purpose, so the folding rule is testable on its
own rather than only inside a Qt build.
"""

import threading

# How often a repeated identical failure is allowed back into the log.
REPEAT_EVERY = 25

_counts: dict = {}
_lock = threading.Lock()


def signature_of(error, fallback: str = "") -> str:
    """What makes two failures "the same" for folding purposes.

    The exception type and HTTP status separate "the provider is down" from "your key is
    wrong", which need entirely different reactions and must never be folded together.
    """
    if error is None:
        return "message|{}".format(str(fallback)[:160])
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(error, "code", None)
    return "{}|{}|{}".format(
        type(error).__name__, "" if status is None else status, str(error)[:120]
    )


def record(error, fallback: str = "") -> int:
    """Count one occurrence and return its running total in the current window."""
    key = signature_of(error, fallback)
    with _lock:
        count = _counts.get(key, 0) + 1
        _counts[key] = count
    return count


def should_surface(count: int) -> bool:
    """Whether the count-th occurrence is worth a line in the log.

    The first one carries the traceback, so it always surfaces; after that only every
    REPEAT_EVERY-th does, which keeps a long outage visible without flooding the log.
    """
    return count == 1 or count % REPEAT_EVERY == 0


def reset() -> None:
    """Start a fresh window so a run's summary covers that run."""
    with _lock:
        _counts.clear()


def summary(limit: int = 8) -> dict:
    """What was folded away, most frequent first: {"total": n, "groups": [...]}."""
    with _lock:
        groups = sorted(_counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "total": sum(count for _, count in groups),
        "groups": [{"signature": sig, "count": n} for sig, n in groups[:limit]],
    }
