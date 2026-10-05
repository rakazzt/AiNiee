import json
import unittest
from pathlib import Path

from ModuleFolders.Infrastructure.LLMRequester.ThinkingProfiles import (
    PROFILE_DASHSCOPE,
    PROFILE_DEFAULT,
    PROFILE_DEEPSEEK,
    PROFILE_OPENAI,
    PROFILE_OPENROUTER,
    PROFILE_VOLCENGINE,
    PROFILE_XAI,
    PROFILE_ZHIPU,
    PROFILES,
    build_thinking_params,
    resolve_profile_name,
)

PRESET_PATH = Path(__file__).parents[1] / "Resource" / "platforms" / "preset.json"

NON_OPENAI_TAG_PREFIXES = ("sakura", "LocalLLM", "google", "anthropic", "amazonbedrock")


class ResolveProfileNameTests(unittest.TestCase):
    """profile 选择是这次重构的核心：数据说用哪个，代码负责怎么算。"""

    def test_declared_profile_wins_over_sniffing(self):
        # URL 看着像 OpenRouter，但数据明确声明用 deepseek，就必须听数据的
        config = {
            "target_platform": "custom_platform_1",
            "api_url": "https://openrouter.ai/api/v1",
            "profile": "deepseek",
        }
        self.assertEqual(resolve_profile_name(config), PROFILE_DEEPSEEK)

    def test_preset_key_is_used_when_profile_is_absent(self):
        config = {"target_platform": "whatever_1", "preset_key": "volcengine"}
        self.assertEqual(resolve_profile_name(config), PROFILE_VOLCENGINE)

    def test_unknown_declared_profile_falls_through(self):
        config = {"target_platform": "openrouter_9", "profile": "not-a-profile"}
        self.assertEqual(resolve_profile_name(config), PROFILE_OPENROUTER)

    def test_legacy_sniffing_still_covers_old_configs(self):
        cases = {
            "openai": PROFILE_OPENAI,
            "deepseek_482913": PROFILE_DEEPSEEK,
            "xai": PROFILE_XAI,
            "volcengine_1": PROFILE_VOLCENGINE,
            "zhipu": PROFILE_ZHIPU,
            "dashscope_7": PROFILE_DASHSCOPE,
            "openrouter_5": PROFILE_OPENROUTER,
        }
        for tag, expected in cases.items():
            with self.subTest(tag=tag):
                self.assertEqual(resolve_profile_name({"target_platform": tag}), expected)

    def test_legacy_sniffing_by_url(self):
        cases = {
            "https://api.deepseek.com/v1": PROFILE_DEEPSEEK,
            "https://api.x.ai/v1": PROFILE_XAI,
            "https://openrouter.ai/api/v1": PROFILE_OPENROUTER,
            "https://open.bigmodel.cn/api/paas/v4": PROFILE_ZHIPU,
            "https://dashscope.aliyuncs.com/compatible-mode/v1": PROFILE_DASHSCOPE,
            "https://ark.cn-beijing.volces.com/api/v3": PROFILE_VOLCENGINE,
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                self.assertEqual(resolve_profile_name({"api_url": url}), expected)

    def test_volcengine_still_takes_precedence(self):
        # 迁移前 volcengine 最先判定并直接返回，顺序不能变
        config = {"target_platform": "custom_1", "api_url": "https://ark.cn-beijing.volces.com/api/v3"}
        self.assertEqual(resolve_profile_name(config), PROFILE_VOLCENGINE)

    def test_unknown_platform_falls_back_to_default(self):
        self.assertEqual(resolve_profile_name({"target_platform": "mystery_1"}), PROFILE_DEFAULT)

    def test_empty_config_is_safe(self):
        self.assertEqual(resolve_profile_name({}), PROFILE_DEFAULT)
        self.assertEqual(resolve_profile_name(None), PROFILE_DEFAULT)


class ProfileBehaviourTests(unittest.TestCase):
    """挑几个关键方言，确认注册表真的在生效（全量行为由表征基线守住）。"""

    def _build(self, profile, **config):
        base = {"model": "m", "messages": [], "extra_body": {}}
        return build_thinking_params(base, dict({"profile": profile}, **config))

    def test_openrouter_sends_reasoning_effort_not_the_shorthand(self):
        params = self._build(PROFILE_OPENROUTER, think_switch=True, think_depth="high")
        self.assertEqual(params["extra_body"]["reasoning"], {"effort": "high"})
        self.assertNotIn("reasoning_effort", params)

    def test_openrouter_disables_with_effort_none(self):
        params = self._build(PROFILE_OPENROUTER, think_switch=False, think_depth="high")
        self.assertEqual(params["extra_body"]["reasoning"], {"effort": "none"})

    def test_deepseek_maps_depth_to_two_levels(self):
        for depth, expected in (("low", "high"), ("high", "high"), ("xhigh", "max"), ("max", "max")):
            with self.subTest(depth=depth):
                params = self._build(PROFILE_DEEPSEEK, think_switch=True, think_depth=depth)
                self.assertEqual(params["reasoning_effort"], expected)

    def test_xai_ignores_the_switch(self):
        # Grok 的推理关不掉，迁移前后都始终发送
        on = self._build(PROFILE_XAI, think_switch=True, think_depth="low")
        off = self._build(PROFILE_XAI, think_switch=False, think_depth="low")
        self.assertEqual(on["reasoning_effort"], "low")
        self.assertEqual(off["reasoning_effort"], "low")

    def test_openai_reasoning_model_drops_temperature(self):
        params = build_thinking_params(
            {"model": "gpt-5.6-luna", "messages": [], "temperature": 0.7, "extra_body": {}},
            {"profile": PROFILE_OPENAI, "think_switch": True, "think_depth": "high",
             "model_name": "gpt-5.6-luna"},
        )
        self.assertEqual(params["reasoning_effort"], "high")
        self.assertNotIn("temperature", params)

    def test_dashscope_passes_thinking_budget(self):
        params = self._build(PROFILE_DASHSCOPE, think_switch=True, thinking_budget=2048)
        self.assertEqual(params["extra_body"]["enable_thinking"], True)
        self.assertEqual(params["extra_body"]["thinking_budget"], 2048)

    def test_unknown_profile_name_is_treated_as_default(self):
        params = build_thinking_params(
            {"model": "m", "messages": [], "extra_body": {}},
            {"profile": "gone", "think_switch": True, "think_depth": "high"},
        )
        # resolve 会先落回嗅探/默认，最终走 default 分支
        self.assertEqual(params["reasoning_effort"], "high")


class RealPresetProfileTests(unittest.TestCase):
    """真实 preset.json 的自检：profile 名字必须是注册表认识的。"""

    @classmethod
    def setUpClass(cls):
        cls.preset = json.loads(PRESET_PATH.read_text(encoding="utf-8"))

    def test_every_declared_profile_exists_in_the_registry(self):
        problems = []
        for tag, platform in self.preset["platforms"].items():
            profile = platform.get("profile")
            if profile is not None and profile not in PROFILES:
                problems.append(f"{tag}: {profile!r}")
        self.assertEqual(problems, [], "unknown profile names in preset.json")

    def test_every_openai_routed_platform_declares_a_profile(self):
        """声明了 profile 就不再依赖字符串嗅探；漏一个就会退回兜底路径。"""
        missing = []
        for tag, platform in self.preset["platforms"].items():
            if tag.startswith(NON_OPENAI_TAG_PREFIXES):
                continue
            if platform.get("profile") not in PROFILES:
                missing.append(tag)
        self.assertEqual(missing, [], "platforms routed to OpenaiRequester without a profile")


if __name__ == "__main__":
    unittest.main()
