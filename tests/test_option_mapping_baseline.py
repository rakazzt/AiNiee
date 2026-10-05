import json
import unittest
from pathlib import Path

from ModuleFolders.Infrastructure.LLMRequester.LLMRequester import is_decision_platform
from ModuleFolders.Infrastructure.LLMRequester.OpenaiRequester import OpenaiRequester

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "option_mapping_baseline.json"
PRESET_PATH = Path(__file__).parents[1] / "Resource" / "platforms" / "preset.json"

# LLMRequester.sent_request 只把这几个 tag 交给 OpenaiRequester 处理
NON_OPENAI_TAG_PREFIXES = ("sakura", "LocalLLM", "google", "anthropic", "amazonbedrock")


class OptionMappingBaselineTests(unittest.TestCase):
    """表征测试：冻结重构前 apply_platform_thinking_params 的逐字节行为。

    基线同时冻结了输入与输出，所以改 preset.json 不会误伤它。
    重构后出现任何 diff 都必须是有意为之并写明理由；无声差异即为回归。
    """

    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        cls.requester = OpenaiRequester()

    def _fresh(self, value):
        return json.loads(json.dumps(value))

    def test_every_frozen_case_still_produces_identical_params(self):
        cases = self.fixture["cases"]
        self.assertTrue(cases, "baseline fixture is empty")

        mismatches = []
        for case_id, case in sorted(cases.items()):
            actual = self.requester.apply_platform_thinking_params(
                self._fresh(case["base"]), self._fresh(case["config"])
            )
            if actual != case["expected"]:
                mismatches.append((case_id, case["expected"], actual))

        if mismatches:
            detail = "\n".join(
                f"  {case_id}\n     expected: {expected}\n     actual  : {actual}"
                for case_id, expected, actual in mismatches[:8]
            )
            self.fail(
                f"{len(mismatches)} of {len(cases)} frozen cases changed behaviour:\n{detail}"
            )

    def test_platform_instance_suffix_does_not_change_behaviour(self):
        """生成的实例 key（openai_482913）必须与裸 tag 走同一分支。

        身份判定在今天用的是 startswith，这条锁定该性质，避免重构改成等值比较。
        """
        cases = self.fixture["cases"]
        mismatched = []
        for case_id, case in cases.items():
            key_form = case_id.split("|")[0]
            if "_482913" not in key_form:
                continue
            bare_id = case_id.replace(key_form, key_form.split("_482913")[0])
            if bare_id not in cases:
                continue
            if case["expected"] != cases[bare_id]["expected"]:
                mismatched.append(case_id)
        self.assertEqual(mismatched, [], "generated platform keys diverged from bare tags")

    def test_baseline_covers_every_platform_routed_to_openai(self):
        preset = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
        routed = sorted(
            tag
            for tag, platform in preset["platforms"].items()
            if not tag.startswith(NON_OPENAI_TAG_PREFIXES)
            # 决策模型接口由 sent_request 提前拦下，从不进入聊天请求器，所以不需要基线用例。
            # 判定复用生产代码的同一个函数，避免两边各写一份而走偏。
            and not is_decision_platform(platform)
        )
        covered = sorted({case_id.split("|")[0].split("_")[0] for case_id in self.fixture["cases"]})
        self.assertEqual(
            covered,
            routed,
            "preset platforms changed but the baseline fixture was not regenerated",
        )


if __name__ == "__main__":
    unittest.main()
