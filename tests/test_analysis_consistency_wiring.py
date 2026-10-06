"""AnalysisTask wiring: the sweep verdicts must actually reach the grouping rule.

The sweep itself is covered by test_consistency_sweep.py, which runs everywhere. What is left
to prove is the glue inside the task: that candidate sources are handed over in the shape the
sweep expects, that a verdict really blocks a merge, and that nothing changes when no decision
model is configured or the switch is off.

AnalysisTask imports the Qt base, so this module SKIPS locally rather than pretend; CI installs
the real dependencies and runs it. The skip is narrow: an ImportError from any module that is
not a known heavy dependency is re-raised, so a real import bug here cannot hide as a skip.
"""
import unittest
from types import SimpleNamespace
from unittest import mock
from urllib.error import URLError

from ModuleFolders.Infrastructure.DecisionEngine import ConsistencySweep as sweep_module
from ModuleFolders.Infrastructure.DecisionEngine import SystemOneClient
from tests.test_consistency_sweep import ARTHUR, FENGSHEN, FENGSHEN_ZHAN, make_engine, replies

HEAVY_ROOTS = {
    "PyQt5", "qfluentwidgets", "rapidjson", "openai", "anthropic", "boto3", "botocore",
    "google", "httpx", "curl_cffi", "tiktoken", "rich", "chardet", "bs4", "mediapipe",
    "regex", "langcodes",
}


def _load():
    try:
        from ModuleFolders.Service.TaskExecutor.AnalysisTask import AnalysisTask
        return AnalysisTask, ""
    except ImportError as error:
        missing = getattr(error, "name", "") or ""
        if missing.split(".")[0] not in HEAVY_ROOTS:
            raise
        return None, "not installed here: {}".format(missing)


AnalysisTask, SKIP_REASON = _load()


def make_task(engine, sweep_switch=True):
    """A real AnalysisTask with only the state these methods touch."""
    task = AnalysisTask.__new__(AnalysisTask)
    task.config = SimpleNamespace(extract_consistency_sweep_switch=sweep_switch)
    task._decision_engine = engine
    task._decision_engine_resolved = True
    task._last_source_sweep = None
    task.logs = []
    task.info = task.logs.append
    task.warning = task.logs.append
    task.load_config = lambda: {"platforms": {}}
    return task


def raw_inputs(*sources):
    return {
        source: {"source": source, "merged_sources": [source],
                 "candidates": [{"candidate_source": source, "type": "term",
                                "recommended_translation": "T:" + source}]}
        for source in sources
    }


@unittest.skipUnless(AnalysisTask is not None, SKIP_REASON)
class TestApplyConsistencySweep(unittest.TestCase):
    def apply(self, task, raw, **kwargs):
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen",
                               side_effect=replies(**kwargs)):
            return task._apply_consistency_sweep(raw)

    def test_a_derived_name_is_reported_as_keep_separate(self):
        task = make_task(make_engine())
        raw = raw_inputs(FENGSHEN, FENGSHEN_ZHAN)
        kept, keep_separate = self.apply(
            task, raw, relations={(FENGSHEN, FENGSHEN_ZHAN): sweep_module.SUBORDINATE})
        self.assertEqual(keep_separate, {(FENGSHEN, FENGSHEN_ZHAN)})
        self.assertEqual(sorted(kept), sorted([FENGSHEN, FENGSHEN_ZHAN]))

    def test_a_generic_term_is_removed_from_the_candidates(self):
        task = make_task(make_engine())
        raw = raw_inputs(ARTHUR, "\u5927\u4eba")
        kept, _ = self.apply(task, raw, generic={"\u5927\u4eba": 0.9})
        self.assertNotIn("\u5927\u4eba", kept)
        self.assertIn(ARTHUR, kept)
        self.assertTrue(any("剔除" in line for line in task.logs))

    def test_a_single_character_candidate_never_becomes_a_group(self):
        task = make_task(make_engine())
        kept, _ = self.apply(task, raw_inputs("\u5251", ARTHUR))
        self.assertNotIn("\u5251", kept)

    def test_no_decision_model_leaves_everything_untouched(self):
        task = make_task(None)
        raw = raw_inputs(FENGSHEN, FENGSHEN_ZHAN)
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen") as post:
            kept, keep_separate = task._apply_consistency_sweep(raw)
        self.assertIs(kept, raw)
        self.assertEqual(keep_separate, set())
        post.assert_not_called()

    def test_the_switch_off_leaves_everything_untouched(self):
        task = make_task(make_engine(), sweep_switch=False)
        raw = raw_inputs(FENGSHEN, FENGSHEN_ZHAN)
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen") as post:
            kept, keep_separate = task._apply_consistency_sweep(raw)
        self.assertIs(kept, raw)
        self.assertEqual(keep_separate, set())
        post.assert_not_called()

    def test_a_model_outage_leaves_the_groups_alone(self):
        task = make_task(make_engine())
        raw = raw_inputs(FENGSHEN, FENGSHEN_ZHAN)
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen",
                               side_effect=URLError("down")):
            kept, keep_separate = task._apply_consistency_sweep(raw)
        self.assertEqual(keep_separate, set())
        self.assertEqual(sorted(kept), sorted([FENGSHEN, FENGSHEN_ZHAN]))

    def test_an_empty_candidate_table_is_returned_as_is(self):
        task = make_task(make_engine())
        kept, keep_separate = task._apply_consistency_sweep({})
        self.assertEqual((kept, keep_separate), ({}, set()))


@unittest.skipUnless(AnalysisTask is not None, SKIP_REASON)
class TestTermConsistencyReport(unittest.TestCase):
    def task_with(self, subordinate):
        task = make_task(make_engine())
        task._last_source_sweep = SimpleNamespace(subordinate=list(subordinate))
        return task

    def final_data(self):
        return {
            "characters": [{"source": FENGSHEN, "recommended_translation": "Wind God", "note": ""}],
            "terms": [{"source": FENGSHEN_ZHAN, "recommended_translation": "Divine Wind Slash", "note": ""}],
        }

    def test_an_inconsistent_pair_is_logged_and_noted_on_the_longer_term(self):
        task = self.task_with([(FENGSHEN, FENGSHEN_ZHAN)])
        final_data = self.final_data()
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen",
                               side_effect=replies(consistency={(FENGSHEN, FENGSHEN_ZHAN): 0.03})):
            task._sweep_term_consistency(final_data)
        self.assertTrue(any(line.startswith("术语一致性：") for line in task.logs))
        self.assertIn(FENGSHEN, final_data["terms"][0]["note"])

    def test_a_consistent_pair_is_left_alone(self):
        task = self.task_with([(FENGSHEN, FENGSHEN_ZHAN)])
        final_data = self.final_data()
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen",
                               side_effect=replies(consistency={(FENGSHEN, FENGSHEN_ZHAN): 0.95})):
            task._sweep_term_consistency(final_data)
        self.assertEqual(final_data["terms"][0]["note"], "")
        # The summary line still reports "found 0", so assert on the per-pair warning only.
        self.assertFalse(any(line.startswith("术语一致性：") for line in task.logs))

    def test_no_subordinate_pairs_means_no_request(self):
        task = self.task_with([])
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen") as post:
            task._sweep_term_consistency(self.final_data())
        post.assert_not_called()

    def test_a_task_that_never_swept_sources_does_nothing(self):
        task = make_task(make_engine())
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen") as post:
            task._sweep_term_consistency(self.final_data())
        post.assert_not_called()




@unittest.skipUnless(AnalysisTask is not None, SKIP_REASON)
class TestSecondStageRowContract(unittest.TestCase):
    """The decision prompt may return one row per distinct entity inside a group.

    That is only safe if the collection step handles the extra row. When the group was merged
    (no decision model, so no verdict), the alias map folds the short name into the long one,
    and the second row must be dropped - not duplicated, and not crash.
    """

    def task_with(self, merged):
        task = make_task(make_engine())
        if merged:
            task.grouped_stage_two_inputs = {
                FENGSHEN_ZHAN: {
                    "source": FENGSHEN_ZHAN,
                    "merged_sources": [FENGSHEN_ZHAN, FENGSHEN],
                    "candidates": [
                        {"candidate_source": FENGSHEN_ZHAN, "type": "term"},
                        {"candidate_source": FENGSHEN, "type": "term"},
                    ],
                },
            }
            task.grouped_stage_two_source_aliases = {
                FENGSHEN_ZHAN: FENGSHEN_ZHAN, FENGSHEN: FENGSHEN_ZHAN,
            }
        else:
            task.grouped_stage_two_inputs = {
                source: {"source": source, "merged_sources": [source],
                         "candidates": [{"candidate_source": source, "type": "term"}]}
                for source in (FENGSHEN_ZHAN, FENGSHEN)
            }
            task.grouped_stage_two_source_aliases = {FENGSHEN_ZHAN: FENGSHEN_ZHAN,
                                                     FENGSHEN: FENGSHEN}
        return task

    ROWS = [
        {"source": FENGSHEN_ZHAN, "recommended_translation": "Wind God Slash", "category_path": ""},
        {"source": FENGSHEN, "recommended_translation": "Wind God", "category_path": ""},
    ]

    def test_an_extra_row_for_a_merged_group_is_dropped_not_duplicated(self):
        final = self.task_with(merged=True)._finalize_results([], [{"terms": list(self.ROWS)}])
        self.assertEqual([row["source"] for row in final["terms"]], [FENGSHEN_ZHAN])
        self.assertEqual(final["terms"][0]["recommended_translation"], "Wind God Slash")

    def test_both_rows_survive_when_the_pair_was_kept_apart(self):
        final = self.task_with(merged=False)._finalize_results([], [{"terms": list(self.ROWS)}])
        self.assertEqual(sorted(row["source"] for row in final["terms"]),
                         sorted([FENGSHEN, FENGSHEN_ZHAN]))

    def test_a_row_for_a_source_the_group_never_had_is_still_accepted(self):
        """The no-invention guard lives in the prompt, not here - but it must not crash."""
        rows = [{"source": ARTHUR, "recommended_translation": "King Arthur"}]
        final = self.task_with(merged=True)._finalize_results([], [{"terms": rows}])
        self.assertIn(ARTHUR, [row["source"] for row in final["terms"]])


if __name__ == "__main__":
    unittest.main()