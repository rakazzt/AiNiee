"""Regression cases for upstream PR #1093's response-prefix fix."""
import contextlib
import io
import unittest

from ModuleFolders.Domain.ResponseChecker.BaseChecks import check_dict_order
from ModuleFolders.Domain.ResponseExtractor.ResponseExtractor import ResponseExtractor


class ResponsePrefixRegressionTests(unittest.TestCase):
    def setUp(self):
        self.extractor = ResponseExtractor()

    def test_transport_prefix_does_not_eat_body_numbers(self):
        self.assertEqual(
            self.extractor.remove_numbered_prefix({"0": "1.1.3.14 is pi", "1": "2.1.5倍"}),
            {"0": "1.3.14 is pi", "1": "1.5倍"},
        )

    def test_multiline_prefix_does_not_replace_version_in_body(self):
        self.assertEqual(
            self.extractor.remove_numbered_prefix({"0": "1.1.3.,版本1.3.4发布"}),
            {"0": "版本1.3.4发布"},
        )

    def test_extraction_error_has_dict_contract(self):
        # 这条会走到 ResponseExtractor 的异常分支，那里会 print 中文告警。
        # 测试不能依赖控制台编码（CI 的 Windows runner 是 cp1252，直接打会
        # UnicodeEncodeError），所以把 stdout 重定向掉。
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.extractor.text_extraction({"0": "source"}, None), {})

    def test_order_accepts_quoted_prefix_but_rejects_nonstring(self):
        source = {"0": "first", "1": "second"}
        self.assertTrue(check_dict_order(source, {"0": '" 1.first', "1": " 2.second"}))
        self.assertFalse(check_dict_order(source, {"0": 1, "1": "2.second"}))


if __name__ == "__main__":
    unittest.main()
