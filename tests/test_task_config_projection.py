import json
import unittest
from pathlib import Path
from unittest.mock import patch

from ModuleFolders.Infrastructure.TaskConfig.TaskConfig import TaskConfig

PRESET_PATH = Path(__file__).parents[1] / "Resource" / "platforms" / "preset.json"


class TaskConfigProjectionTests(unittest.TestCase):
    """B2 的验收：声明的选项必须真的出现在 TaskConfig 产出的请求参数里。

    设计评审指出，之前的测试直接调 apply_options_to_params，**绕过了这一层**，
    而断链恰恰发生在这里 —— get_active_platform_configuration 是一张固定 16 键的
    白名单，UI 存了值但不在名单里就永远到不了请求体，而且不会有任何报错
    （OpenRouter 会静默忽略未知键）。所以这条测试必须走真实的投影层。
    """

    def setUp(self):
        self.preset = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
        # 只测投影接线，preset 读取本身另有测试；这样本测试不依赖 rapidjson
        patcher = patch.object(TaskConfig, "load_platform_presets", return_value=self.preset)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _params_for(self, platform_key, platform_data):
        task_config = TaskConfig()
        task_config.platforms = {platform_key: platform_data}
        task_config.interface_role = "active"
        # 预置这两项可跳过 prepare_for_active_platform，避免依赖配置文件状态
        task_config.target_platform = platform_key
        task_config.prepared_interface_role = "active"
        task_config.base_url = platform_data.get("api_url", "")
        task_config.model = platform_data.get("model")
        task_config.request_timeout = 60

        with patch.object(TaskConfig, "get_active_platform_tag", return_value=platform_key):
            return task_config.get_active_platform_configuration()

    def test_declared_option_reaches_the_request_params(self):
        params = self._params_for("openrouter_482913", {
            "tag": "openrouter_482913",
            "api_url": "https://openrouter.ai/api/v1",
            "api_format": "OpenAI",
            "model": "google/gemini-3.8-flash",
            "provider_order": ["Anthropic", "Google"],
        })
        self.assertEqual(params["extra_body"]["provider"]["order"], ["Anthropic", "Google"])

    def test_several_options_share_one_parent_object(self):
        params = self._params_for("openrouter_482913", {
            "tag": "openrouter_482913",
            "api_format": "OpenAI",
            "provider_order": ["Anthropic"],
            "provider_zdr": True,
            "fallback_models": ["google/gemini-3.8-flash"],
        })
        self.assertEqual(params["extra_body"]["provider"]["order"], ["Anthropic"])
        self.assertEqual(params["extra_body"]["provider"]["zdr"], True)
        self.assertEqual(params["extra_body"]["models"], ["google/gemini-3.8-flash"])

    def test_unset_options_are_not_sent(self):
        params = self._params_for("openrouter_482913", {
            "tag": "openrouter_482913",
            "api_format": "OpenAI",
        })
        self.assertEqual(params["extra_body"], {})

    def test_platform_without_options_is_unaffected(self):
        params = self._params_for("zhipu_111111", {
            "tag": "zhipu_111111",
            "api_format": "OpenAI",
            "think_switch": True,
            "think_depth": "high",
        })
        self.assertEqual(params["extra_body"], {})
        # 旧的白名单字段必须原样保留
        self.assertEqual(params["think_switch"], True)
        self.assertEqual(params["think_depth"], "high")
        self.assertEqual(params["api_format"], "OpenAI")

    def test_legacy_projection_keys_survive(self):
        params = self._params_for("openrouter_482913", {
            "tag": "openrouter_482913",
            "api_url": "https://openrouter.ai/api/v1",
            "api_format": "OpenAI",
            "model": "google/gemini-3.8-flash",
            "region": "", "access_key": "", "secret_key": "",
            "temperature": 1.0, "tls_switch": False,
            "think_switch": False, "think_depth": "medium",
            "thinking_budget": -1, "thinking_level": "high",
            "provider_order": ["Anthropic"],
        })
        for key in (
            "target_platform", "api_url", "api_key", "api_format", "model_name",
            "region", "access_key", "secret_key", "request_timeout", "temperature",
            "extra_body", "tls_switch", "think_switch", "think_depth",
            "thinking_budget", "thinking_level",
        ):
            self.assertIn(key, params, f"legacy projection key {key} disappeared")

    def test_stored_extra_body_is_not_mutated_by_the_projection(self):
        shared = {"provider": {"order": ["Google"]}, "custom": 1}
        platform_data = {
            "tag": "openrouter_482913",
            "api_format": "OpenAI",
            "provider_order": ["Anthropic"],
            "extra_body": shared,
        }
        params = self._params_for("openrouter_482913", platform_data)

        self.assertEqual(params["extra_body"]["provider"]["order"], ["Anthropic"])
        # 用户存的那份必须原封不动，否则下一次 save_config 会把合并结果持久化
        self.assertEqual(shared, {"provider": {"order": ["Google"]}, "custom": 1})
        self.assertEqual(platform_data["extra_body"], {"provider": {"order": ["Google"]}, "custom": 1})
        self.assertEqual(params["extra_body"]["custom"], 1)

    def test_max_tokens_reaches_the_requester(self):
        """PR 1090 把 Sakura 的 max_tokens 从硬编码 512 改成读平台配置。

        投影层不带上这个键，那条配置路径就永远是死的（只能回落 512），
        和「UI 存了值但请求里没有」是同一类断链。
        """
        params = self._params_for("sakura_111111", {
            "tag": "sakura_111111",
            "api_format": "OpenAI",
            "max_tokens": 2048,
        })
        self.assertEqual(params["max_tokens"], 2048)

    def test_max_tokens_is_none_when_unset(self):
        # 未配置时为 None，请求器按 `or 512` 回落，行为与改造前一致
        params = self._params_for("zhipu_111111", {"tag": "zhipu_111111"})
        self.assertIsNone(params["max_tokens"])


if __name__ == "__main__":
    unittest.main()
