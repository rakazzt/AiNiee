import json
import unittest
from pathlib import Path

from ModuleFolders.Infrastructure.LLMRequester.OptionSchema import (
    OptionSchemaError,
    apply_options_to_params,
    build_option_body,
    decode_option_value,
    encode_option_value,
    localized_text,
    matches_when,
    merge_into,
    option_defaults,
    option_widget_plan,
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


class ApplyOptionsToParamsTests(unittest.TestCase):
    """TaskConfig 每轮请求都会走这里：声明式选项必须真的进到请求参数里。"""

    def _params(self):
        # 形状与 TaskConfig.get_active_platform_configuration 的产出一致
        return {
            "target_platform": "openrouter_777",
            "api_url": "https://openrouter.ai/api/v1",
            "model_name": "google/gemini-3.8-flash",
            "temperature": 1.0,
            "extra_body": {},
            "think_switch": True,
            "think_depth": "medium",
        }

    def test_platform_without_options_is_byte_identical(self):
        """回归：没有声明 options 的平台，params 一个字节都不能变。"""
        params = self._params()
        before = json.dumps(params, sort_keys=True)
        apply_options_to_params(params, {"tag": "zhipu"}, PRESET)
        self.assertEqual(json.dumps(params, sort_keys=True), before)

    def test_declared_options_reach_the_request(self):
        params = self._params()
        apply_options_to_params(
            params,
            {"tag": "openrouter_777", "provider_order": ["Anthropic"],
             "allow_fallbacks": False},
            PRESET,
        )
        self.assertEqual(params["extra_body"]["provider"]["order"], ["Anthropic"])
        self.assertEqual(params["extra_body"]["provider"]["allow_fallbacks"], False)

    def test_unset_options_add_nothing(self):
        params = self._params()
        apply_options_to_params(params, {"tag": "openrouter_777"}, PRESET)
        self.assertEqual(params["extra_body"], {})

    def test_user_extra_body_siblings_survive(self):
        # extra_body 打底、选项值按叶子覆盖：同层其它键必须保留
        params = self._params()
        params["extra_body"] = {"provider": {"allow_fallbacks": True}, "custom_thing": 1}
        apply_options_to_params(
            params,
            {"tag": "openrouter_777", "provider_order": ["Anthropic"]},
            PRESET,
        )
        self.assertEqual(params["extra_body"]["provider"]["allow_fallbacks"], True)
        self.assertEqual(params["extra_body"]["provider"]["order"], ["Anthropic"])
        self.assertEqual(params["extra_body"]["custom_thing"], 1)

    def test_option_value_overrides_extra_body_leaf(self):
        params = self._params()
        params["extra_body"] = {"provider": {"order": ["Google"]}}
        apply_options_to_params(
            params, {"tag": "openrouter_777", "provider_order": ["Anthropic"]}, PRESET
        )
        self.assertEqual(params["extra_body"]["provider"]["order"], ["Anthropic"])

    def test_input_config_is_not_mutated(self):
        """params["extra_body"] 是 config 里的对象引用，合并绝不能写回 config。"""
        shared = {"provider": {"order": ["Google"]}}
        platform_config = {"tag": "openrouter_777", "provider_order": ["Anthropic"],
                           "extra_body": shared}
        params = self._params()
        params["extra_body"] = shared

        apply_options_to_params(params, platform_config, PRESET)

        self.assertEqual(shared, {"provider": {"order": ["Google"]}},
                         "the user's stored extra_body was modified in place")
        self.assertEqual(platform_config["extra_body"], {"provider": {"order": ["Google"]}})

    def test_param_vehicle_lands_top_level(self):
        preset = {"p": {"options": [{"key": "effort", "type": "string",
                                     "vehicle": "param", "path": "reasoning_effort"}]}}
        params = self._params()
        apply_options_to_params(params, {"tag": "p", "effort": "high"}, preset)
        self.assertEqual(params["reasoning_effort"], "high")

    def test_extra_body_is_always_a_dict_afterwards(self):
        # 平台没配 extra_body 时不能把 None 留在参数里
        params = self._params()
        params["extra_body"] = None
        apply_options_to_params(params, {"tag": "zhipu"}, PRESET)
        self.assertIsInstance(params["extra_body"], dict)


class MergeIntoTests(unittest.TestCase):
    def test_overlay_wins_and_siblings_survive(self):
        target = {"a": 1, "nested": {"x": 1, "y": 2}}
        merge_into(target, {"b": 2, "nested": {"y": 9}})
        self.assertEqual(target, {"a": 1, "b": 2, "nested": {"x": 1, "y": 9}})

    def test_nested_dict_replaces_scalar(self):
        target = {"a": 1}
        merge_into(target, {"a": {"deep": True}})
        self.assertEqual(target, {"a": {"deep": True}})

    def test_empty_overlay_is_a_no_op(self):
        target = {"a": 1}
        merge_into(target, {})
        merge_into(target, None)
        self.assertEqual(target, {"a": 1})


class RealPresetSchemaTests(unittest.TestCase):
    """对真实 preset.json 的自检。

    preset.json 是被鼓励 drop-in 替换的数据文件，能绕过 CI 生效，所以这份校验
    在运行期也必须存在；这里把它固化成测试，任何声明写错都会在 CI 里红。
    """

    @classmethod
    def setUpClass(cls):
        preset_path = Path(__file__).parents[1] / "Resource" / "platforms" / "preset.json"
        cls.preset = json.loads(preset_path.read_text(encoding="utf-8"))

    def test_every_declared_schema_is_valid(self):
        problems = []
        for tag, platform in self.preset["platforms"].items():
            options = platform.get("options")
            if options is None:
                continue
            for error in validate_schema(options):
                problems.append(f"{tag}: {error}")
        self.assertEqual(problems, [])

    def test_openrouter_declares_the_full_routing_surface(self):
        options = {o["key"] for o in self.preset["platforms"]["openrouter"]["options"]}
        expected = {
            "provider_order", "provider_only", "provider_ignore", "provider_sort",
            "provider_quantizations", "provider_allow_fallbacks", "provider_require_parameters",
            "provider_data_collection", "provider_zdr", "provider_enforce_distillable_text",
            "provider_max_price_prompt", "provider_max_price_completion",
            "provider_preferred_max_latency", "provider_preferred_min_throughput",
            "fallback_models", "reasoning_summary",
        }
        self.assertEqual(expected - options, set(), "missing routing options")

    def test_option_keys_do_not_collide_with_legacy_settings(self):
        """选项键不能与 key_in_settings 撞名。

        否则界面会出现两组写同一个配置键的控件（一套走旧的手写方法、一套走
        schema），后写的赢，用户会看到重复项。
        """
        for tag, platform in self.preset["platforms"].items():
            legacy = set(platform.get("key_in_settings") or [])
            for descriptor in platform.get("options") or []:
                self.assertNotIn(
                    descriptor["key"], legacy,
                    f"{tag}: option {descriptor['key']} collides with a legacy setting key",
                )

    def test_openrouter_routing_options_do_not_touch_reasoning_effort(self):
        """reasoning.effort 目前仍由 OpenaiRequester 的 legacy 分支写。

        schema 里若也声明它就会双写同一个叶子，值可能互相覆盖。等渲染器迁移
        完成后再搬过来，并同步更新表征基线。
        """
        paths = {o["path"] for o in self.preset["platforms"]["openrouter"]["options"]}
        self.assertNotIn("reasoning.effort", paths)


class LocalizedTextTests(unittest.TestCase):
    def test_plain_string_passes_through(self):
        self.assertEqual(localized_text("Hello", "简中", "fb"), "Hello")

    def test_language_map_picks_the_language(self):
        value = {"简中": "提供商顺序", "English": "Provider order"}
        self.assertEqual(localized_text(value, "简中", "fb"), "提供商顺序")
        self.assertEqual(localized_text(value, "English", "fb"), "Provider order")

    def test_missing_language_falls_back_to_any_translation(self):
        self.assertEqual(localized_text({"English": "Only"}, "简中", "fb"), "Only")

    def test_unusable_value_returns_the_fallback(self):
        self.assertEqual(localized_text(None, "简中", "fallback"), "fallback")
        self.assertEqual(localized_text({}, "简中", "fallback"), "fallback")


class OptionWidgetPlanTests(unittest.TestCase):
    def test_type_maps_to_widget_kind(self):
        cases = {
            "bool": "switch", "enum": "combo", "string": "line", "int": "line",
            "float": "line", "string-list": "list", "json": "json",
            "number-or-object": "json",
        }
        for option_type, expected in cases.items():
            with self.subTest(option_type=option_type):
                descriptor = {"key": "k", "type": option_type, "values": ["a"]}
                self.assertEqual(option_widget_plan(descriptor)["widget"], expected)

    def test_combo_offers_an_unset_choice_first(self):
        # 没有「不设置」这一项，用户就无法把值改回服务端默认
        plan = option_widget_plan({"key": "k", "type": "enum", "values": ["a", "b"]})
        self.assertEqual(plan["choices"], ["", "a", "b"])

    def test_label_falls_back_to_the_key(self):
        plan = option_widget_plan({"key": "provider_order", "type": "string"})
        self.assertEqual(plan["label"], "provider_order")

    def test_label_is_localized(self):
        descriptor = {"key": "k", "type": "string",
                      "label": {"简中": "提供商顺序", "English": "Provider order"}}
        self.assertEqual(option_widget_plan(descriptor, "English")["label"], "Provider order")

    def test_numeric_flag_is_set_for_number_types(self):
        self.assertTrue(option_widget_plan({"key": "k", "type": "int"})["numeric"])
        self.assertFalse(option_widget_plan({"key": "k", "type": "string"})["numeric"])


class EncodeOptionValueTests(unittest.TestCase):
    def _d(self, option_type, **extra):
        return dict({"key": "k", "type": option_type}, **extra)

    def test_empty_text_means_unset(self):
        for option_type in ("string", "int", "string-list", "json"):
            with self.subTest(option_type=option_type):
                self.assertEqual(encode_option_value(self._d(option_type), "   "), (True, None))

    def test_string_list_accepts_newlines_and_commas(self):
        ok, value = encode_option_value(self._d("string-list"), "Anthropic\nGoogle, DeepSeek")
        self.assertTrue(ok)
        self.assertEqual(value, ["Anthropic", "Google", "DeepSeek"])

    def test_int_parses(self):
        self.assertEqual(encode_option_value(self._d("int"), "42"), (True, 42))

    def test_bad_int_is_rejected(self):
        self.assertEqual(encode_option_value(self._d("int"), "4x2"), (False, None))

    def test_float_parses(self):
        self.assertEqual(encode_option_value(self._d("float"), "1.5"), (True, 1.5))

    def test_nan_and_infinity_are_rejected(self):
        """nan/inf 会让 rapidjson 写出的 config 读不回来。

        Config.load_config 解析失败时会把 config 改名 .corrupt 并清空 ——
        用户的全部平台与密钥会在应用内消失，所以必须在这里挡住。
        """
        for text in ("nan", "NaN", "inf", "-inf", "Infinity"):
            with self.subTest(text=text):
                ok, _ = encode_option_value(self._d("float"), text)
                self.assertFalse(ok, f"{text} must not reach the config")

    def test_json_object_parses(self):
        ok, value = encode_option_value(self._d("json"), '{"a": 1}')
        self.assertTrue(ok)
        self.assertEqual(value, {"a": 1})

    def test_malformed_json_is_rejected(self):
        self.assertEqual(encode_option_value(self._d("json"), "{oops"), (False, None))

    def test_string_is_kept_verbatim_including_apostrophes(self):
        ok, value = encode_option_value(self._d("string"), "it's ok")
        self.assertTrue(ok)
        self.assertEqual(value, "it's ok")


class DecodeOptionValueTests(unittest.TestCase):
    def test_none_is_empty_text(self):
        self.assertEqual(decode_option_value({"type": "string"}, None), "")

    def test_list_round_trips_as_newlines(self):
        self.assertEqual(
            decode_option_value({"type": "string-list"}, ["a", "b"]), "a\nb"
        )

    def test_object_round_trips_as_json(self):
        self.assertEqual(
            decode_option_value({"type": "json"}, {"a": 1}), '{"a": 1}'
        )

    def test_bool_round_trips(self):
        self.assertEqual(decode_option_value({"type": "bool"}, True), "true")
        self.assertEqual(decode_option_value({"type": "bool"}, False), "false")

    def test_round_trip_through_the_ui_helpers(self):
        descriptor = {"key": "k", "type": "string-list"}
        ok, value = encode_option_value(descriptor, "a\nb")
        self.assertTrue(ok)
        self.assertEqual(decode_option_value(descriptor, value), "a\nb")


if __name__ == "__main__":
    unittest.main()
