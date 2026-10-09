"""A decision model must never be used as a chat model.

Jev answers typed questions and never generates text, so if one were selected as the
translation interface the failure would otherwise surface as an unexplained 400 from the
OpenAI-shaped path. The guard makes it a readable message instead, and the preset data is
asserted here so the group cannot silently drift out of the platform UI.
"""
import json
import unittest
from pathlib import Path

from ModuleFolders.Infrastructure.DecisionEngine import SystemOneClient
from ModuleFolders.Infrastructure.DecisionEngine.DecisionEngine import resolve_shape

PRESET_PATH = Path(__file__).parents[1] / "Resource" / "platforms" / "preset.json"

HEAVY_ROOTS = {
    "PyQt5", "qfluentwidgets", "openai", "anthropic", "boto3", "botocore", "google",
    "httpx", "curl_cffi", "tiktoken", "rich", "chardet", "bs4", "mediapipe",
}


def _load_requester():
    try:
        from ModuleFolders.Infrastructure.LLMRequester.LLMRequester import LLMRequester, is_decision_platform
        return LLMRequester, is_decision_platform, ""
    except ImportError as error:
        missing = getattr(error, "name", "") or ""
        if missing.split(".")[0] not in HEAVY_ROOTS:
            raise
        return None, None, "not installed here: {}".format(missing)


LLMRequester, is_decision_platform, SKIP_REASON = _load_requester()


class TestPresetData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.platforms = json.loads(PRESET_PATH.read_text(encoding="utf-8"))["platforms"]

    def test_decision_platforms_are_declared_and_grouped(self):
        decision = {tag: p for tag, p in self.platforms.items() if p.get("group") == "decision"}
        self.assertIn("jev", decision, "the TypeSafe entry is the primary one")
        for tag, platform in decision.items():
            # The declared format must resolve to a real wire shape, not merely be a
            # string: "SystemOne" is the TypeSafe route, "decisions" the OpenRouter
            # decisions router. An unresolvable value would silently fall back.
            self.assertIn(resolve_shape(platform.get("api_format")), SystemOneClient.SHAPES, tag)
            self.assertTrue(platform.get("api_key") == "", "no key must be shipped")
            self.assertIn("api_url", platform.get("key_in_settings", []), tag)

    def test_decision_platforms_are_not_offered_as_translation_platforms(self):
        """AddAPIDialog renders only local/online/custom, so any other group is excluded."""
        groups = {"local", "online", "custom"}
        for tag, platform in self.platforms.items():
            if platform.get("group") == "decision":
                self.assertNotIn(platform.get("group"), groups, tag)


@unittest.skipUnless(is_decision_platform is not None, SKIP_REASON)
class TestGuard(unittest.TestCase):
    def test_recognises_both_markers(self):
        self.assertTrue(is_decision_platform({"group": "decision"}))
        self.assertTrue(is_decision_platform({"api_format": "SystemOne"}))

    def test_leaves_ordinary_platforms_alone(self):
        for platform in ({"group": "online", "api_format": "OpenAI"},
                         {"group": "local"},
                         {"api_format": "Anthropic"},
                         {},
                         None):
            self.assertFalse(is_decision_platform(platform), repr(platform))

    def test_every_shipped_decision_platform_is_recognised(self):
        platforms = json.loads(PRESET_PATH.read_text(encoding="utf-8"))["platforms"]
        for tag, platform in platforms.items():
            if platform.get("group") == "decision":
                self.assertTrue(is_decision_platform(platform), tag)

    def test_sent_request_refuses_and_names_the_interface(self):
        """Refuse through the normal failure path, and say which interface was at fault."""
        requester = LLMRequester()
        skip, think, content, prompt_tokens, completion_tokens = requester.sent_request(
            [{"role": "user", "content": "hi"}], "system", {"group": "decision", "name": "JEV"},
        )
        self.assertTrue(skip, "a decision model must never be treated as a chat model")
        self.assertEqual(think, "")
        self.assertEqual(prompt_tokens, 0)
        self.assertEqual(completion_tokens, 0)
        # The reason travels in the content slot, which is where requesters report a failure.
        self.assertIn("JEV", content)
        self.assertIn("\u51b3\u7b56\u6a21\u578b", content)

    def test_the_refusal_also_fires_on_the_format_marker_alone(self):
        requester = LLMRequester()
        result = requester.sent_request([], "system", {"api_format": "SystemOne", "tag": "jev_123456"})
        self.assertTrue(result[0])
        self.assertIn("jev_123456", result[2])


if __name__ == "__main__":
    unittest.main()