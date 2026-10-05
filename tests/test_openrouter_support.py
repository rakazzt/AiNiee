import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from ModuleFolders.Infrastructure.LLMRequester.OpenaiRequester import OpenaiRequester


class OpenRouterReasoningParamTests(unittest.TestCase):
    """OpenRouter 用原生 reasoning.effort；它与顶层 reasoning_effort 互斥，只能发一个。"""

    def setUp(self):
        self.requester = OpenaiRequester()

    def _params(self, think_switch, think_depth, extra_body=None):
        config = {
            "target_platform": "openrouter",
            "api_url": "https://openrouter.ai/api/v1",
            "model_name": "google/gemini-3.8-flash",
            "think_switch": think_switch,
            "think_depth": think_depth,
        }
        base = {"model": "google/gemini-3.8-flash", "messages": []}
        if extra_body is not None:
            base["extra_body"] = extra_body
        return self.requester.apply_platform_thinking_params(base, config)

    def test_effort_follows_think_depth(self):
        # 界面档位与 OpenRouter 的 effort 取值一一对应，直接透传
        for depth in ("low", "medium", "high", "xhigh", "max"):
            params = self._params(True, depth)
            self.assertEqual(params["extra_body"]["reasoning"], {"effort": depth})
            # 两者同时出现且不同会被 OpenRouter 拒绝
            self.assertNotIn("reasoning_effort", params)
            self.assertNotIn("reasoning_effort", params["extra_body"])

    def test_disabled_think_sends_effort_none(self):
        params = self._params(False, "high")
        self.assertEqual(params["extra_body"]["reasoning"], {"effort": "none"})

    def test_unknown_depth_falls_back_to_medium(self):
        params = self._params(True, "bogus")
        self.assertEqual(params["extra_body"]["reasoning"], {"effort": "medium"})

    def test_user_reasoning_options_are_preserved(self):
        # extra_body 里用户自己写的附加项不能被覆盖掉
        params = self._params(True, "high", extra_body={"reasoning": {"summary": "concise"}})
        self.assertEqual(params["extra_body"]["reasoning"], {"summary": "concise", "effort": "high"})

    def test_legacy_reasoning_effort_is_scrubbed(self):
        params = self._params(True, "high", extra_body={"reasoning_effort": "low"})
        self.assertNotIn("reasoning_effort", params["extra_body"])

    def test_other_openai_compatible_platform_is_unchanged(self):
        # 回归：通用平台仍然走顶层 reasoning_effort 分支
        config = {
            "target_platform": "custom_platform_1",
            "api_url": "https://example.invalid/v1",
            "model_name": "gpt-4o",
            "think_switch": True,
            "think_depth": "high",
        }
        params = self.requester.apply_platform_thinking_params({"model": "gpt-4o", "messages": []}, config)
        self.assertEqual(params["reasoning_effort"], "high")
        self.assertNotIn("reasoning", params.get("extra_body", {}))

    def test_provider_routing_survives_untouched(self):
        # provider 路由不是专门功能，只能靠 extra_body 透传，这里锁定它不被思考参数逻辑破坏
        routing = {
            "order": ["Anthropic", "Google"],
            "allow_fallbacks": False,
            "data_collection": "deny",
            "zdr": True,
            "sort": "throughput",
        }
        params = self._params(True, "high", extra_body={"provider": routing})
        self.assertEqual(params["extra_body"]["provider"], routing)
        self.assertEqual(params["extra_body"]["reasoning"], {"effort": "high"})

    def test_model_fallback_list_survives(self):
        params = self._params(
            True,
            "medium",
            extra_body={"models": ["anthropic/claude-sonnet-5.5", "google/gemini-3.8-flash"]},
        )
        self.assertEqual(
            params["extra_body"]["models"],
            ["anthropic/claude-sonnet-5.5", "google/gemini-3.8-flash"],
        )


class OpenRouterReasoningExtractionTests(unittest.TestCase):
    """OpenRouter 把推理放在 reasoning 字段，DeepSeek 放在 reasoning_content。"""

    def setUp(self):
        self.requester = OpenaiRequester()

    def test_reasoning_field_is_extracted(self):
        message = SimpleNamespace(content="译文", reasoning="先想想")
        self.assertEqual(self.requester._extract_reasoning_text(message), "先想想")

    def test_reasoning_content_still_supported(self):
        message = SimpleNamespace(content="译文", reasoning_content="deepseek 思考")
        self.assertEqual(self.requester._extract_reasoning_text(message), "deepseek 思考")

    def test_reasoning_details_are_joined(self):
        message = SimpleNamespace(
            content="译文",
            reasoning=None,
            reasoning_details=[
                {"type": "reasoning.text", "text": "第一段"},
                {"type": "reasoning.summary", "summary": "第二段"},
            ],
        )
        self.assertEqual(self.requester._extract_reasoning_text(message), "第一段第二段")

    def test_reasoning_details_ignores_empty_entries(self):
        message = SimpleNamespace(content="译文", reasoning=None, reasoning_details=[{"type": "reasoning.encrypted", "data": "x"}])
        self.assertEqual(self.requester._extract_reasoning_text(message), "")

    def test_null_content_does_not_crash(self):
        # 纯推理模型可能只返回 reasoning，content 为 null
        response = MagicMock()
        response.choices[0].message = SimpleNamespace(content=None, reasoning="只有思考")
        think, content, prompt_tokens, completion_tokens = self.requester._extract_from_completion(response)
        self.assertEqual(think, "只有思考")
        self.assertEqual(content, "")

    def test_reasoning_only_content_still_splits_on_think_tag(self):
        response = MagicMock()
        response.choices[0].message = SimpleNamespace(content="<think>想</think>译文", reasoning=None)
        think, content, _, _ = self.requester._extract_from_completion(response)
        self.assertEqual(think, "想")
        self.assertEqual(content, "译文")

    def test_sse_delta_reasoning_is_merged(self):
        raw = (
            'data: {"choices":[{"delta":{"reasoning":"想一"}}]}\n'
            'data: {"choices":[{"delta":{"reasoning":"想二"}}]}\n'
            'data: {"choices":[{"delta":{"content":"译文"}}],"usage":{"prompt_tokens":7,"completion_tokens":3}}\n'
            "data: [DONE]\n"
        )
        think, content, prompt_tokens, completion_tokens = self.requester._parse_sse_response(raw)
        self.assertEqual(think, "想一想二")
        self.assertEqual(content, "译文")
        self.assertEqual(prompt_tokens, 7)
        self.assertEqual(completion_tokens, 3)


if __name__ == "__main__":
    unittest.main()
