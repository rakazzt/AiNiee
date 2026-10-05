"""前缀缓存（prompt caching）的回归测试。

两件事必须钉住：

1. **稳定前缀真的稳定。** 缓存只做严格前缀匹配，所以按批次变化的内容（术语表 /
   禁翻表 / 项目表）必须全部落在断点之后。切错位置比不缓存更贵：Anthropic 会退化成
   「每轮都写缓存、永远读不到」，而缓存写入按 1.25 倍计费。
2. **usage 口径正确。** Anthropic 的 ``input_tokens`` 不含缓存读写，直接当
   prompt_tokens 用会让开了缓存的统计反而变小 —— 看着像省钱，其实是少算。
"""
import unittest
from types import SimpleNamespace

from ModuleFolders.Domain.PromptBuilder.PromptBuilder import PromptBuilder
from ModuleFolders.Domain.PromptBuilder.PromptBuilderEnum import PromptBuilderEnum
from ModuleFolders.Infrastructure.LLMRequester import PromptCache
from ModuleFolders.Infrastructure.LLMRequester.AnthropicRequester import AnthropicRequester
from ModuleFolders.Infrastructure.LLMRequester.LLMClientFactory import LLMClientFactory
from ModuleFolders.Infrastructure.LLMRequester.OpenaiRequester import OpenaiRequester

GLOSSARY_ALICE = {"src": "Alice", "dst": "爱丽丝", "info": ""}
GLOSSARY_BOB = {"src": "Bob", "dst": "鲍勃", "info": ""}


def _config(**overrides):
    config = SimpleNamespace(
        target_language="chinese_simplified",
        translation_prompt_selection={
            "last_selected_id": PromptBuilderEnum.COMMON,
            "prompt_content": "自定义提示词",
        },
        prompt_dictionary_switch=False,
        prompt_dictionary_data=[],
        prompt_dictionary_match_case_sensitive=False,
        prompt_dictionary_match_whole_word=False,
        exclusion_list_switch=False,
        exclusion_list_data=[],
        translation_example_switch=False,
        translation_example_data=[],
        characterization_switch=False,
        world_building_switch=False,
        writing_style_switch=False,
        project_characters_data=[],
        project_terms_data=[],
        project_non_translate_data=[],
        few_shot_and_example_switch=False,
        pre_line_counts=0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _generate(config, source_text_dict, source_lang="japanese"):
    messages, system, extra_log, system_stable = PromptBuilder.generate_prompt(
        config, source_text_dict, [], source_lang
    )
    return messages, system, extra_log, system_stable


class StablePrefixSplitTests(unittest.TestCase):
    """断点必须落在「整份任务逐字节不变」的那一段之后。"""

    def _full_config(self, **overrides):
        base = dict(
            prompt_dictionary_switch=True,
            prompt_dictionary_data=[GLOSSARY_ALICE, GLOSSARY_BOB],
            world_building_switch=True,
            world_building_content="王国坐落在北境。",
            writing_style_switch=True,
            writing_style_content="保持冷峻的叙述口吻。",
            translation_example_switch=True,
            translation_example_data=[{"src": "你好", "dst": "こんにちは"}],
            project_characters_data=[
                {"source": "Alice", "recommended_translation": "爱丽丝", "gender": "女", "note": "主角"},
                {"source": "Bob", "recommended_translation": "鲍勃", "gender": "男", "note": ""},
            ],
        )
        base.update(overrides)
        return _config(**base)

    def test_cacheable_prefix_is_identical_across_batches(self):
        """同一个任务的两个批次，稳定前缀必须逐字节相同 —— 这就是缓存能命中的前提。"""
        config = self._full_config()

        _, system_a, _, stable_a = _generate(config, {"0": "Alice went to the castle."})
        _, system_b, _, stable_b = _generate(config, {"0": "Bob stayed at home."})

        # 系统提示词整体确实变了（术语表按批次筛选）
        self.assertNotEqual(system_a, system_b)
        # 但可缓存的那一段完全没变
        self.assertEqual(stable_a, stable_b)
        self.assertTrue(system_a.startswith(stable_a))
        self.assertTrue(system_b.startswith(stable_b))

    def test_per_batch_content_never_lands_in_the_stable_prefix(self):
        config = self._full_config()
        _, system, _, stable = _generate(config, {"0": "Alice went to the castle."})

        # 稳定前缀里只有整份任务不变的内容
        self.assertIn("王国坐落在北境。", stable)
        self.assertIn("保持冷峻的叙述口吻。", stable)
        self.assertIn("こんにちは", stable)

        # 按批次筛选出来的内容一律在断点之后
        self.assertIn("爱丽丝", system)
        self.assertNotIn("爱丽丝", stable)
        self.assertIn("###角色表", system)
        self.assertNotIn("###角色表", stable)

    def test_no_content_is_lost_or_duplicated(self):
        """切分不能改变送出去的提示词内容，只能改变它在哪一侧。"""
        config = self._full_config()
        source_text_dict = {"0": "Alice went to the castle."}
        _, system, _, stable = _generate(config, source_text_dict)

        expected = (
            PromptBuilder.build_system(config, "japanese")
            + PromptBuilder.build_world_building(config)
            + PromptBuilder.build_writing_style(config)
            + PromptBuilder.build_translation_example(config)
            + PromptBuilder.build_glossary_prompt(config, source_text_dict)
            + PromptBuilder.build_project_characters_prompt(config, source_text_dict)
        )
        self.assertEqual(system, expected)
        self.assertEqual(system[len(stable):], expected[len(stable):])

    def test_without_optional_blocks_the_whole_system_is_cacheable(self):
        """默认配置（无术语表/背景/文风/示例）下整个系统提示词都是稳定前缀。"""
        config = _config()
        _, system, _, stable = _generate(config, {"0": "Hello."})
        self.assertEqual(system, stable)

    def test_user_message_is_untouched_by_the_split(self):
        config = self._full_config()
        messages, system, _, stable = _generate(config, {"0": "Alice went to the castle."})
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("Alice went to the castle.", messages[0]["content"])
        self.assertTrue(system.startswith(stable))


class SystemBlockTests(unittest.TestCase):
    def test_only_the_stable_block_carries_the_breakpoint(self):
        blocks = PromptCache.build_system_blocks("STABLE", "STABLE+DYNAMIC")
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["text"], "STABLE")
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(blocks[1]["text"], "+DYNAMIC")
        self.assertNotIn("cache_control", blocks[1])

    def test_wholly_stable_prompt_yields_a_single_marked_block(self):
        blocks = PromptCache.build_system_blocks("ALL", "ALL")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral"})

    def test_missing_or_mismatched_prefix_is_refused(self):
        # 宁可不缓存，也不能把断点切在错误的位置上
        self.assertEqual(PromptCache.build_system_blocks("", "DYNAMIC"), [])
        self.assertEqual(PromptCache.build_system_blocks("   ", "DYNAMIC"), [])
        self.assertEqual(PromptCache.build_system_blocks("OTHER", "DYNAMIC"), [])


class ExplicitCacheGatingTests(unittest.TestCase):
    def test_anthropic_format_gets_explicit_breakpoints(self):
        self.assertTrue(PromptCache.supports_explicit_cache({"api_format": "Anthropic"}))

    def test_openrouter_gets_explicit_breakpoints(self):
        self.assertTrue(
            PromptCache.supports_explicit_cache(
                {"api_format": "OpenAI", "api_url": "https://openrouter.ai/api/v1"}
            )
        )
        self.assertTrue(
            PromptCache.supports_explicit_cache({"target_platform": "openrouter_123"})
        )

    def test_gemini_through_openrouter_keeps_the_plain_string(self):
        """Gemini 把缓存的 systemInstruction 当不可变：稳定块 + 每批筛选的术语表
        写在同一条 system 消息里，术语表会被一起固化而失效，所以退回单字符串。"""
        self.assertFalse(
            PromptCache.supports_explicit_cache(
                {
                    "api_format": "OpenAI",
                    "target_platform": "openrouter",
                    "model_name": "google/gemini-2.5-pro",
                }
            )
        )

    def test_auto_cache_providers_are_left_alone(self):
        """OpenAI / DeepSeek / 自定义中转都是自动前缀缓存：多送字段没有收益，
        还可能被严格校验字段的中转站直接拒收。"""
        for platform_config in (
            {"api_format": "OpenAI", "api_url": "https://api.openai.com/v1", "target_platform": "openai"},
            {"api_format": "OpenAI", "api_url": "https://api.deepseek.com", "target_platform": "deepseek"},
            {"api_format": "Google", "api_url": "https://generativelanguage.googleapis.com"},
            {"api_format": "OpenAI", "api_url": "https://my-gateway.example.com/v1", "target_platform": "custom_1"},
            None,
        ):
            self.assertFalse(PromptCache.supports_explicit_cache(platform_config), platform_config)


class UsageAccountingTests(unittest.TestCase):
    def test_anthropic_totals_include_cache_read_and_write(self):
        """input_tokens 只是断点之后那一段，漏掉缓存读写会让统计反向变小。"""
        usage = SimpleNamespace(
            input_tokens=50,
            cache_read_input_tokens=1000,
            cache_creation_input_tokens=200,
        )
        read, write, total = PromptCache.anthropic_usage_totals(usage)
        self.assertEqual((read, write, total), (1000, 200, 1250))
        self.assertGreater(total, usage.input_tokens)

    def test_anthropic_totals_without_cache_are_unchanged(self):
        usage = SimpleNamespace(input_tokens=321)
        self.assertEqual(PromptCache.anthropic_usage_totals(usage), (0, 0, 321))
        self.assertEqual(PromptCache.anthropic_usage_totals(None), (0, 0, 0))

    def test_openai_cache_hit_tokens_covers_both_field_shapes(self):
        # OpenAI / OpenRouter
        self.assertEqual(
            PromptCache.openai_cache_hit_tokens(
                {"prompt_tokens": 1500, "prompt_tokens_details": {"cached_tokens": 1024}}
            ),
            1024,
        )
        self.assertEqual(
            PromptCache.openai_cache_hit_tokens(
                SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=768))
            ),
            768,
        )
        # DeepSeek
        self.assertEqual(
            PromptCache.openai_cache_hit_tokens(
                {"prompt_tokens": 1500, "prompt_cache_hit_tokens": 1408}
            ),
            1408,
        )
        # 没有缓存字段时不能抛错，也不能凭空造出命中
        self.assertEqual(PromptCache.openai_cache_hit_tokens({"prompt_tokens": 10}), 0)
        self.assertEqual(PromptCache.openai_cache_hit_tokens(None), 0)

    def test_usage_summary_distinguishes_no_cache_from_no_data(self):
        PromptCache.reset_usage()
        PromptCache.record_usage(0, 0, 500)
        self.assertIn("缓存命中 0 Tokens", PromptCache.format_usage())

        PromptCache.reset_usage()
        PromptCache.record_usage(1024, 0, 1024)
        summary = PromptCache.format_usage()
        self.assertIn("缓存命中 1024 Tokens", summary)
        self.assertIn("命中率 50.0%", summary)

        snapshot = PromptCache.usage_snapshot()
        self.assertEqual(snapshot["read"], 1024)
        self.assertEqual(snapshot["requests"], 1)


STABLE = "### 基础提示词\n逐行翻译。"
DYNAMIC = "\n###术语表\nAlice|爱丽丝|"


def _patch_client(attr, fake_client):
    """Replace a factory client getter; returns a restore callable."""
    original = getattr(LLMClientFactory, attr)
    setattr(LLMClientFactory, attr, lambda self, platform_config: fake_client)
    return lambda: setattr(LLMClientFactory, attr, original)


class OpenaiRequesterWiringTests(unittest.TestCase):
    """断点必须真的出现在发出去的报文里，否则前面的逻辑都白搭。"""

    def setUp(self):
        PromptCache.reset_usage()

    def _capture_system_message(self, platform_config):
        captured = []

        def fake_create(**params):
            captured.append(params)
            # 解析失败会走 SSE 兜底分支，但参数已经抓到了
            return SimpleNamespace(parse=lambda: None, text="")

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    with_raw_response=SimpleNamespace(create=fake_create)
                )
            )
        )
        restore = _patch_client("get_openai_client", fake_client)
        try:
            OpenaiRequester().request_openai(
                [{"role": "user", "content": "hello"}],
                STABLE + DYNAMIC,
                platform_config,
                STABLE,
            )
        finally:
            restore()
        return captured[0]["messages"][0]

    def test_openrouter_system_message_carries_the_breakpoint(self):
        message = self._capture_system_message(
            {"api_format": "OpenAI", "target_platform": "openrouter", "model_name": "anthropic/claude-sonnet-4.5"}
        )
        self.assertEqual(message["role"], "system")
        blocks = message["content"]
        self.assertIsInstance(blocks, list)
        self.assertEqual(blocks[0]["text"], STABLE)
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(blocks[1]["text"], DYNAMIC)
        self.assertNotIn("cache_control", blocks[1])

    def test_auto_cache_provider_keeps_the_plain_string(self):
        """DeepSeek 是自动前缀缓存：多送 cache_control 没有收益，还可能被拒收。"""
        message = self._capture_system_message(
            {"api_format": "OpenAI", "api_url": "https://api.deepseek.com", "model_name": "deepseek-chat"}
        )
        self.assertEqual(message["content"], STABLE + DYNAMIC)


class AnthropicRequesterWiringTests(unittest.TestCase):
    def setUp(self):
        PromptCache.reset_usage()

    def _call(self, platform_config, usage):
        captured = []
        response = SimpleNamespace(content=[], stop_reason="end_turn", usage=usage)

        def fake_create(**params):
            captured.append(params)
            return response

        restore = _patch_client(
            "get_anthropic_client", SimpleNamespace(messages=SimpleNamespace(create=fake_create))
        )
        try:
            result = AnthropicRequester().request_anthropic(
                [{"role": "user", "content": "hi"}],
                STABLE + DYNAMIC,
                platform_config,
                STABLE,
            )
        finally:
            restore()
        return result, captured[0]

    def test_breakpoint_is_sent_and_cache_tokens_are_counted(self):
        usage = SimpleNamespace(
            input_tokens=50,
            output_tokens=7,
            cache_read_input_tokens=1000,
            cache_creation_input_tokens=200,
        )
        result, params = self._call(
            {"api_format": "Anthropic", "model_name": "claude-sonnet-5"}, usage
        )

        system = params["system"]
        self.assertIsInstance(system, list)
        self.assertEqual(system[0]["text"], STABLE)
        self.assertEqual(system[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(system[1]["text"], DYNAMIC)

        # 50 只是断点之后那一段；总量必须是 读+写+未命中
        self.assertEqual(result[3], 1250)
        self.assertEqual(result[4], 7)

    def test_without_cache_tokens_the_total_is_unchanged(self):
        usage = SimpleNamespace(input_tokens=321, output_tokens=12)
        result, _ = self._call(
            {"api_format": "Anthropic", "model_name": "claude-sonnet-5"}, usage
        )
        self.assertEqual(result[3], 321)
        self.assertEqual(result[4], 12)


if __name__ == "__main__":
    unittest.main()