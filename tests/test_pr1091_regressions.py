"""Focused regressions for upstream PR #1091's analysis-task guard.

The whole module needs PyQt5 through Base, so this runs natively in CI and under
a stubbed harness locally. _validate_base_structure is a pure method, so it is
called unbound with None as self.
"""
import unittest

from ModuleFolders.Service.TaskExecutor.AnalysisTask import AnalysisTask

validate = AnalysisTask._validate_base_structure


class AnalysisStructureGuardTests(unittest.TestCase):
    def test_wellformed_payload_passes(self):
        ok, message = validate(None, {"characters": [{"source": "A"}], "terms": []},
                               ("characters", "terms"))
        self.assertTrue(ok, message)

    def test_string_elements_are_rejected(self):
        """模型把数组回成 ["Alice"] 时下游 row.get 会 AttributeError 并中断整个分析。

        校验必须拦住它，让该块走重试路径。
        """
        ok, message = validate(None, {"characters": ["Alice"], "terms": []},
                               ("characters", "terms"))
        self.assertFalse(ok)
        self.assertIn("characters", message)

    def test_missing_and_wrong_type_fields_are_rejected(self):
        self.assertFalse(validate(None, {"terms": []}, ("characters", "terms"))[0])
        self.assertFalse(validate(None, {"characters": {}, "terms": []},
                                  ("characters", "terms"))[0])


if __name__ == "__main__":
    unittest.main()
