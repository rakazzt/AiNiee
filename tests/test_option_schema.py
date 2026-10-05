import unittest

from ModuleFolders.Infrastructure.LLMRequester.OptionSchema import (
    OptionSchemaError,
    build_option_body,
    matches_when,
    option_defaults,
    resolve_options,
    resolve_preset_key,
    validate_schema,
    write_path,
)

PRESET = {
    "openrouter": {
        "tag": "openrouter",
        "api_url": "https://openrouter.ai/api/v1",
        "options": [
            {
                "key": "provider_order",
                "type": "string-list",
                "vehicle": "extra_body",
                "path": "provider.order",
                "default": [],
            },
            {
                "key": "allow_fallbacks",
                "type": "bool",
                "vehicle": "extra_body",
                "path": "provider.allow_fallbacks",
            },
            {
                "key": "reasoning_effort",
                "type": "enum",
                "values": ["low", "medium", "high", "none"],
                "default": "none",
                "vehicle": "extra_body",
                "path": "reasoning.effort",
                "when": [{"option": "allow_fallbacks", "truthy": True}],
            },
        ],
    },
    "zhipu": {"tag": "zhipu", "api_url": "https://open.bigmodel.cn/api/paas/v4"},
}


class ResolvePresetKeyTests(unittest.TestCase):
    """schema 必须能挂到存量平台实例上，否则老用户看不到任何新选项。"""

    def test_explicit_preset_key_wins(self):
        config = {"tag": "whatever_1", "preset_key": "openrouter"}
        self.assertEqual(resolve_preset_key(config, PRESET), "openrouter")

    def test_bare_tag_resolves(self):
        self.assertEqual(resolve_preset_key({"tag": "openrouter"}, PRESET), "openrouter")

    def test_generated_instance_key_resolves_to_its_preset(self):
        # PlatformPage 写成 f"{tag}_{random}"
        self.assertEqual(
            resolve_preset_key({"tag": "openrouter_482913"}, PRESET), "openrouter"
        )

    def test_unknown_platform_returns_none(self):
        self.assertIsNone(resolve_preset_key({"tag": "nope_123456"}, PRESET))

    def test_prefix_neighbour_does_not_bleed(self):
        # openrouterx_1 不能被当成 openrouter
        self.assertIsNone(resolve_preset_key({"tag": "openrouterx_1"}, PRESET))

    def test_empty_config_is_safe(self):
        self.assertIsNone(resolve_preset_key({}, PRESET))
        self.assertIsNone(resolve_preset_key(None, PRESET))


class ResolveOptionsTests(unittest.TestCase):
    def test_declared_options_are_returned(self):
        options = resolve_options({"tag": "openrouter_777"}, PRESET)
        self.assertEqual([o["key"] for o in options],
                         ["provider_order", "allow_fallbacks", "reasoning_effort"])

    def test_platform_without_options_returns_empty(self):
        self.assertEqual(resolve_options({"tag": "zhipu"}, PRESET), [])

    def test_unknown_platform_returns_empty(self):
        self.assertEqual(resolve_options({"tag": "ghost"}, PRESET), [])


class ValidateSchemaTests(unittest.TestCase):
    def test_clean_schema_has_no_errors(self):
        self.assertEqual(validate_schema(PRESET["openrouter"]["options"]), [])

    def test_duplicate_key_is_reported(self):
        options = [
            {"key": "a", "type": "bool", "path": "reasoning.effort"},
            {"key": "a", "type": "bool", "path": "provider.sort"},
        ]
        self.assertTrue(any("duplicate" in e for e in validate_schema(options)))

    def test_default_outside_enum_values_is_reported(self):
        options = [{"key": "a", "type": "enum", "values": ["x"], "default": "y",
                    "path": "reasoning.effort"}]
        self.assertTrue(any("default" in e for e in validate_schema(options)))

    def test_when_referencing_unknown_option_is_reported(self):
        options = [{"key": "a", "type": "bool", "path": "reasoning.effort",
                    "when": [{"option": "missing", "eq": True}]}]
        self.assertTrue(any("unknown option" in e for e in validate_schema(options)))

    def test_conflicting_paths_are_reported(self):
        options = [
            {"key": "a", "type": "bool", "path": "provider.order"},
            {"key": "b", "type": "bool", "path": "provider.order"},
        ]
        self.assertTrue(any("already written" in e for e in validate_schema(options)))

    def test_non_list_schema_is_reported(self):
        self.assertEqual(validate_schema({"key": "a"}), ["options must be a list"])


class PathSafetyTests(unittest.TestCase):
    """preset.json 是可以被 drop-in 替换的数据文件，路径必须被代码侧约束。"""

    def _reject(self, path):
        options = [{"key": "evil", "type": "string", "path": path}]
        errors = validate_schema(options)
        self.assertTrue(errors, f"{path} should have been rejected")
        return errors

    def test_reserved_transport_keys_are_rejected(self):
        for path in ("model", "messages", "stream", "timeout", "api_key", "base_url",
                     "headers", "extra_headers"):
            with self.subTest(path=path):
                self._reject(path)

    def test_nested_reserved_key_is_rejected(self):
        self._reject("provider.model")

    def test_unknown_root_is_rejected(self):
        self._reject("secret_backdoor.value")

    def test_illegal_segment_is_rejected(self):
        self._reject("provider.__class__")
        self._reject("provider.a-b")

    def test_too_deep_path_is_rejected(self):
        self._reject("provider.options.anthropic.extra")

    def test_write_path_refuses_to_descend_into_a_scalar(self):
        target = {"provider": "already-a-string"}
        with self.assertRaises(OptionSchemaError):
            write_path(target, "provider.order", ["x"])
        # 原值必须保持不变，不能被静默覆盖
        self.assertEqual(target, {"provider": "already-a-string"})

    def test_write_path_creates_nested_objects(self):
        target = {}
        write_path(target, "provider.order", ["Anthropic"])
        self.assertEqual(target, {"provider": {"order": ["Anthropic"]}})

    def test_leaf_merge_keeps_siblings(self):
        target = {"reasoning": {"summary": "concise"}}
        write_path(target, "reasoning.effort", "high")
        self.assertEqual(target["reasoning"], {"summary": "concise", "effort": "high"})

    def test_parser_is_not_an_evaluator(self):
        # when 是封闭谓词，出现未知谓词一律判否，绝不做表达式求值
        self.assertFalse(matches_when({"when": [{"eval": "__import__('os')"}]}, {}))
        self.assertTrue(matches_when({"when": []}, {}))


class BuildOptionBodyTests(unittest.TestCase):
    def _config(self, **overrides):
        config = {"tag": "openrouter_777", "model": "google/gemini-3.8-flash"}
        config.update(overrides)
        return config

    def test_unset_options_are_omitted_entirely(self):
        body = build_option_body(self._config(), PRESET)
        self.assertEqual(body, {"param": {}, "extra_body": {}})

    def test_empty_values_are_omitted(self):
        for empty in (None, "", []):
            with self.subTest(empty=empty):
                body = build_option_body(self._config(provider_order=empty), PRESET)
                self.assertNotIn("provider", body["extra_body"])

    def test_values_land_on_the_declared_path(self):
        body = build_option_body(
            self._config(provider_order=["Anthropic", "Google"]), PRESET
        )
        self.assertEqual(body["extra_body"]["provider"]["order"],
                         ["Anthropic", "Google"])

    def test_two_options_share_a_parent_object(self):
        body = build_option_body(
            self._config(provider_order=["Anthropic"], allow_fallbacks=False), PRESET
        )
        self.assertEqual(
            body["extra_body"]["provider"],
            {"order": ["Anthropic"], "allow_fallbacks": False},
        )

    def test_predicate_gates_an_option(self):
        # reasoning_effort 只在 allow_fallbacks 为真时才写
        without = build_option_body(self._config(reasoning_effort="high"), PRESET)
        self.assertNotIn("reasoning", without["extra_body"])

        with_gate = build_option_body(
            self._config(reasoning_effort="high", allow_fallbacks=True), PRESET
        )
        self.assertEqual(with_gate["extra_body"]["reasoning"]["effort"], "high")

    def test_map_transforms_the_value(self):
        preset = {
            "p": {"options": [{"key": "depth", "type": "enum", "values": ["low", "high"],
                               "path": "reasoning.effort",
                               "map": {"low": "medium"}}]}
        }
        body = build_option_body({"tag": "p", "depth": "low"}, preset)
        self.assertEqual(body["extra_body"]["reasoning"]["effort"], "medium")

    def test_param_vehicle_writes_top_level(self):
        preset = {"p": {"options": [{"key": "effort", "type": "string",
                                     "vehicle": "param", "path": "reasoning_effort"}]}}
        body = build_option_body({"tag": "p", "effort": "high"}, preset)
        self.assertEqual(body["param"], {"reasoning_effort": "high"})
        self.assertEqual(body["extra_body"], {})

    def test_broken_descriptor_is_skipped_without_raising(self):
        preset = {"p": {"options": [
            {"key": "bad", "type": "string", "path": "messages"},   # reserved
            {"key": "good", "type": "string", "path": "provider.sort"},
        ]}}
        body = build_option_body({"tag": "p", "bad": "x", "good": "throughput"}, preset)
        self.assertNotIn("messages", str(body))
        self.assertEqual(body["extra_body"]["provider"]["sort"], "throughput")

    def test_platform_without_schema_is_inert(self):
        self.assertEqual(build_option_body({"tag": "zhipu"}, PRESET),
                         {"param": {}, "extra_body": {}})


class OptionDefaultsTests(unittest.TestCase):
    def test_defaults_come_only_from_declared_options(self):
        defaults = option_defaults(PRESET["openrouter"]["options"])
        self.assertEqual(defaults, {"provider_order": [], "reasoning_effort": "none"})

    def test_missing_default_is_simply_absent(self):
        self.assertEqual(option_defaults([{"key": "a", "type": "bool"}]), {})


if __name__ == "__main__":
    unittest.main()
