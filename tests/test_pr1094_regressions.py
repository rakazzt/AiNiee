"""Focused regressions for upstream PR #1094's text processing fixes."""
import re
import unittest
from types import SimpleNamespace

from ModuleFolders.Domain.TextProcessor.PolishTextProcessor import PolishTextProcessor
from ModuleFolders.Domain.TextProcessor.TextProcessor import TextProcessor


class TextProcessorRegressionTests(unittest.TestCase):
    def test_literal_windows_path_does_not_raise(self):
        rule = {"src": "foo", "dst": "C:\\game\\", "regex": True}
        processor = TextProcessor(SimpleNamespace(
            pre_translation_data=[rule], post_translation_data=[], exclusion_list_data=[]
        ))
        self.assertEqual(processor.replace_before_translation({"0": "foo"}), {"0": "C:\\game\\"})
        polish = PolishTextProcessor(SimpleNamespace(
            pre_translation_data=[rule], post_translation_data=[]
        ))
        self.assertEqual(polish.replace_before_translation({"0": "foo"}), {"0": "C:\\game\\"})

    def test_zero_width_affix_does_not_loop(self):
        processor = TextProcessor(SimpleNamespace(
            pre_translation_data=[], post_translation_data=[], exclusion_list_data=[]
        ))
        value, _, _ = processor._process_affixes({"0": "hello"}, [re.compile(r"^\b")], [re.compile(r"$")])
        self.assertEqual(value, {"0": "hello"})

    def test_only_preprocessed_numeric_keys_are_recovered(self):
        config = SimpleNamespace(pre_translation_data=[], post_translation_data=[], exclusion_list_data=[])
        for processor in (TextProcessor(config), PolishTextProcessor(config)):
            before = {"0": "1. Title", "1": "【12】正文"}
            pre = processor.digital_sequence_preprocessing(before.copy())
            self.assertEqual(pre["1"], "【12】正文")
            self.assertEqual(processor.digital_sequence_recovery(pre), before)

    def test_br_does_not_match_break_or_broom(self):
        processor = TextProcessor(SimpleNamespace(
            pre_translation_data=[], post_translation_data=[], exclusion_list_data=[]
        ))
        self.assertTrue(processor.RE_BR_TAG.fullmatch("<br/>"))
        self.assertIsNone(processor.RE_BR_TAG.match('<break time="1s"/>'))
        self.assertIsNone(processor.RE_BR_TAG.match("<broom>"))


if __name__ == "__main__":
    unittest.main()
