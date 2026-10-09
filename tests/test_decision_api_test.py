"""The interface test must not chat-test a decision model.

A decision model answers typed questions; it never generates text. Sending it a chat
completion returns 404, which is what a user saw when they pressed 测试 on a GPT-6 Luna
Decisions interface - a confusing dead end for a correctly configured platform.

The cause was two-fold and both halves are pinned here: the test built its own platform
config without the "group" the guard keys on, and the guard's format fallback knew only
"SystemOne", so the "decisions" wire was not recognised either. A decision platform is now
tested the way it is actually used - one real typed question - and never reaches the chat
requesters.

SimpleExecutor pulls the Qt-bound stack, so this skips locally and runs in CI, with the
skip narrowed to known heavy dependencies so a real import bug cannot hide as a skip.
"""
import json
import unittest
from unittest import mock
from urllib.error import HTTPError

from ModuleFolders.Infrastructure.DecisionEngine import SystemOneClient

HEAVY_ROOTS = {
    "PyQt5", "qfluentwidgets", "rapidjson", "openai", "anthropic", "boto3", "botocore",
    "google", "httpx", "curl_cffi", "tiktoken", "rich", "chardet", "bs4", "mediapipe",
    "regex", "langcodes",
}


def _load():
    try:
        from ModuleFolders.Infrastructure.LLMRequester.LLMRequester import LLMRequester
        from ModuleFolders.Service.SimpleExecutor.SimpleExecutor import SimpleExecutor
        return SimpleExecutor, LLMRequester, ""
    except ImportError as error:
        missing = getattr(error, "name", "") or ""
        if missing.split(".")[0] not in HEAVY_ROOTS:
            raise
        return None, None, "not installed here: {}".format(missing)


# Both pull the Qt-bound stack through ModuleFolders.Base, so they load together or not at all.
SimpleExecutor, LLMRequester, SKIP_REASON = _load()

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "openai/gpt-6-luna-decisions"

# Exactly what the platform store holds for the shipped preset, tag renamed as the UI does.
DECISION_PLATFORM = {
    "tag": "luna_decisions_482913",
    "preset_key": "luna_decisions",
    "group": "decision",
    "name": "openrouter luna",
    "api_url": "https://openrouter.ai/api",
    "api_key": "sk-or-v1-test",
    "api_format": "decisions",
    "model": MODEL,
    "auto_complete": False,
}

CHAT_PLATFORM = {
    "tag": "openrouter_482913",
    "preset_key": "openrouter",
    "group": "online",
    "name": "openrouter chat",
    "api_url": "https://openrouter.ai/api/v1",
    "api_key": "sk-or-v1-test",
    "api_format": "OpenAI",
    "model": "openai/gpt-6-luna",
    "auto_complete": True,
}


class FakeResponse:
    def __init__(self, payload=None, raw=None):
        self._raw = raw if raw is not None else json.dumps(payload).encode("utf-8")

    def read(self):
        return self._raw

    def close(self):
        """HTTPError is file-like and closes its fp on dealloc."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def decision_body(probability=0.99):
    return {
        "model": MODEL,
        "answers": {"reachable": {"type": "noul", "noul": probability}},
        "usage": {"input_tokens": 12, "output_tokens": 0, "cost": 0.0000012},
    }


def urlopen(return_value=None, side_effect=None):
    return mock.patch.object(SystemOneClient.urlrequest, "urlopen",
                             return_value=return_value, side_effect=side_effect)


@unittest.skipUnless(SimpleExecutor is not None, SKIP_REASON)
class TestDecisionInterfaceTest(unittest.TestCase):
    def make_executor(self):
        executor = SimpleExecutor.__new__(SimpleExecutor)
        executor.logs = []
        for name in ("info", "error", "warning", "print"):
            setattr(executor, name, executor.logs.append)
        return executor

    def test_a_decision_platform_is_never_sent_to_the_chat_requesters(self):
        """The regression: this path used to end in OpenaiRequester and a 404."""
        executor = self.make_executor()
        with mock.patch.object(LLMRequester, "sent_request") as chat:
            with urlopen(return_value=FakeResponse(decision_body())):
                executor.api_test(None, dict(DECISION_PLATFORM))
        chat.assert_not_called()
        self.assertTrue(any("接口测试成功" in line for line in executor.logs))

    def test_a_decision_platform_is_tested_with_a_real_typed_question(self):
        executor = self.make_executor()
        with urlopen(return_value=FakeResponse(decision_body(0.87))) as post:
            executor.api_test(None, dict(DECISION_PLATFORM))
        payload = json.loads(post.call_args[0][0].data.decode("utf-8"))
        self.assertEqual(payload["model"], MODEL)
        self.assertEqual(sorted(payload["questions"]), ["reachable"])
        self.assertEqual(payload["questions"]["reachable"]["type"], "noul")
        self.assertIn("state", payload)
        self.assertEqual(post.call_args[0][0].full_url, ENDPOINT)

    def test_the_reported_result_names_the_model_and_the_answer(self):
        executor = self.make_executor()
        with urlopen(return_value=FakeResponse(decision_body(0.87))):
            executor.api_test(None, dict(DECISION_PLATFORM))
        report = "\n".join(executor.logs)
        self.assertIn(MODEL, report)
        self.assertIn(ENDPOINT, report)
        self.assertIn("0.87", report)

    def test_an_unreachable_decision_endpoint_fails_with_the_reason(self):
        executor = self.make_executor()
        error = HTTPError(ENDPOINT, 404, "Not Found", {},
                          FakeResponse(raw=b'{"error":{"message":"Not Found","code":404}}'))
        with urlopen(side_effect=error):
            executor.api_test(None, dict(DECISION_PLATFORM))
        report = "\n".join(executor.logs)
        self.assertIn("接口测试失败", report)
        self.assertIn("404", report)

    def test_a_decision_platform_with_no_model_fails_before_any_request(self):
        executor = self.make_executor()
        with urlopen() as post:
            executor.api_test(None, dict(DECISION_PLATFORM, model=""))
        post.assert_not_called()
        self.assertIn("配置有误", "\n".join(executor.logs))

    def test_an_ordinary_chat_platform_still_goes_through_the_chat_requester(self):
        """The decision branch must not have swallowed every other platform."""
        executor = self.make_executor()
        with mock.patch.object(LLMRequester, "sent_request",
                               return_value=(False, "", "hello", 1, 1)) as chat:
            executor.api_test(None, dict(CHAT_PLATFORM))
        chat.assert_called_once()
        self.assertTrue(any("接口测试成功" in line for line in executor.logs))


if __name__ == "__main__":
    unittest.main()
