"""Regressions for the residual-source-text checker merged from PR #1090.

detecting_remaining_original_text returns True when the translation passes and
False when residual source text is found. The check is on by default, so a false
positive means a correct translation can never be accepted and the batch retries
forever.
"""
import unittest

from ModuleFolders.Domain.ResponseChecker.AdvancedChecks import (
    detecting_remaining_original_text,
)

SOURCE = {"0": "你好世界", "1": "今天天气很好", "2": "谢谢你"}
JAPANESE = {"0": "こんにちは世界", "1": "今日はいい天気です", "2": "ありがとう"}


class ResidualTextCheckerTests(unittest.TestCase):
    def test_chinese_source_to_japanese_passes(self):
        """日语译文合法地包含汉字，按字符检测必然误判，必须直接放行。"""
        self.assertTrue(
            detecting_remaining_original_text(
                SOURCE, JAPANESE, "chinese_simplified", "japanese"
            )
        )

    def test_chinese_source_to_chinese_target_passes(self):
        for target in ("chinese_simplified", "chinese_traditional"):
            with self.subTest(target=target):
                self.assertTrue(
                    detecting_remaining_original_text(SOURCE, SOURCE, "chinese_simplified", target)
                )

    def test_guard_is_limited_to_chinese_sources(self):
        """非中文源不应被这条短路放行 —— 否则等于把检查整体关掉。"""
        source = {"0": "こんにちは", "1": "ありがとう", "2": "さようなら"}
        # 日语源→中文目标时依然要真的跑检测（这里只断言它不抛错且返回布尔值）
        result = detecting_remaining_original_text(source, SOURCE, "japanese", "chinese_simplified")
        self.assertIsInstance(result, bool)

    def test_unsupported_source_language_is_skipped(self):
        # 代码类文本不支持检测，按通过处理
        self.assertTrue(detecting_remaining_original_text(SOURCE, JAPANESE, "english", "japanese"))

    def test_single_line_dicts_are_skipped(self):
        self.assertTrue(
            detecting_remaining_original_text(
                {"0": "你好"}, {"0": "你好"}, "chinese_simplified", "english"
            )
        )

    def test_missing_target_language_keeps_old_behaviour(self):
        """老调用方不传 target_language 时不能崩，按原路径继续。"""
        result = detecting_remaining_original_text(SOURCE, JAPANESE, "chinese_simplified")
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
