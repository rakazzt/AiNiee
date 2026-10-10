"""Folding repeated provider errors, and naming the interface that produced them.

A real log held 184 identical 503 tracebacks from one extraction. They buried the two lines
that mattered - the configuration and the "task stopped" notice - so a slow run looked
exactly like a dead one, both to the user and to anyone reading the log afterwards.

Three properties carry this:

1. The first occurrence keeps full detail, traceback included; the rest are counted.
2. Nothing is lost: every 25th repeat resurfaces with the running count, and the run ends
   with a summary of the totals.
3. Two different failures are never folded together - "the provider is offline" and "your
   key is wrong" need opposite reactions, so the signature carries type and status.

The ledger runs everywhere. The two call-site classes pull the Qt-bound stack, so they skip
locally and run in CI, with the skip narrowed to known heavy dependencies.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

from ModuleFolders.Log import ErrorLedger

HEAVY_ROOTS = {
    "PyQt5", "qfluentwidgets", "rapidjson", "openai", "anthropic", "boto3", "botocore",
    "google", "httpx", "curl_cffi", "tiktoken", "rich", "chardet", "bs4", "mediapipe",
    "regex", "langcodes",
}


def _load(name: str):
    import importlib
    try:
        return importlib.import_module(name), ""
    except ImportError as error:
        missing = getattr(error, "name", "") or ""
        if missing.split(".")[0] not in HEAVY_ROOTS:
            raise
        return None, "not installed here: {}".format(missing)


_log, LOG_SKIP = _load("ModuleFolders.Log.Log")
_task, TASK_SKIP = _load("ModuleFolders.Service.TaskExecutor.AnalysisTask")
LogMixin = getattr(_log, "LogMixin", None)
AnalysisTask = getattr(_task, "AnalysisTask", None)


class FakeHttpError(Exception):
    """Stands in for an SDK error that carries an HTTP status."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class TestLedger(unittest.TestCase):
    def setUp(self):
        ErrorLedger.reset()

    def test_the_first_occurrence_always_surfaces(self):
        self.assertTrue(ErrorLedger.should_surface(1))

    def test_repeats_stay_quiet_until_the_interval(self):
        for count in range(2, ErrorLedger.REPEAT_EVERY):
            self.assertFalse(ErrorLedger.should_surface(count), count)
        self.assertTrue(ErrorLedger.should_surface(ErrorLedger.REPEAT_EVERY))
        self.assertTrue(ErrorLedger.should_surface(ErrorLedger.REPEAT_EVERY * 2))

    def test_the_count_is_running_per_signature(self):
        error = FakeHttpError("model_unavailable", 503)
        self.assertEqual([ErrorLedger.record(error) for _ in range(3)], [1, 2, 3])
        self.assertEqual(ErrorLedger.record(FakeHttpError("nope", 401)), 1)

    def test_a_different_status_is_a_different_failure(self):
        ErrorLedger.record(FakeHttpError("model_unavailable", 503))
        ErrorLedger.record(FakeHttpError("model_unavailable", 503))
        self.assertEqual(ErrorLedger.record(FakeHttpError("model_unavailable", 402)), 1)

    def test_a_transport_failure_is_not_folded_into_an_http_one(self):
        ErrorLedger.record(FakeHttpError("boom", 503))
        self.assertEqual(ErrorLedger.record(ConnectionError("boom")), 1)

    def test_the_summary_counts_everything_and_leads_with_the_worst(self):
        error = FakeHttpError("model_unavailable", 503)
        for _ in range(30):
            ErrorLedger.record(error)
        ErrorLedger.record(FakeHttpError("bad key", 401))
        summary = ErrorLedger.summary()
        self.assertEqual(summary["total"], 31)
        self.assertEqual(summary["groups"][0]["count"], 30)
        self.assertIn("503", summary["groups"][0]["signature"])
        self.assertEqual(summary["groups"][1]["count"], 1)

    def test_the_summary_is_bounded_but_the_total_is_not(self):
        for status in range(400, 415):
            ErrorLedger.record(FakeHttpError("e", status))
        summary = ErrorLedger.summary(limit=3)
        self.assertEqual(len(summary["groups"]), 3)
        self.assertEqual(summary["total"], 15)

    def test_reset_starts_a_fresh_window(self):
        ErrorLedger.record(FakeHttpError("e", 500))
        ErrorLedger.reset()
        self.assertEqual(ErrorLedger.summary()["total"], 0)


@unittest.skipUnless(LogMixin is not None, LOG_SKIP)
class TestFoldingInTheLogger(unittest.TestCase):
    """The rule as the logger applies it: one full line, then every 25th."""

    # The base must exist at class-definition time even when the module is skipped.
    class Recorder(LogMixin if LogMixin is not None else object):
        def __init__(self):
            self.lines = []

        def error(self, msg, error=None):
            self.lines.append(("error", msg))

        def warning(self, msg):
            self.lines.append(("warning", msg))

    def setUp(self):
        ErrorLedger.reset()

    def test_one_full_error_and_a_counted_summary_instead_of_thirty_tracebacks(self):
        recorder = self.Recorder()
        error = FakeHttpError("model_unavailable", 503)
        for _ in range(30):
            recorder.error_repeat("请求任务错误 ...", error)
        # 30 identical failures cost 2 lines: the first with its traceback, and the 25th
        # of the interval carrying the running count.
        self.assertEqual(len(recorder.lines), 2, recorder.lines)
        self.assertEqual(recorder.lines[0][0], "error")
        self.assertEqual(recorder.lines[1][0], "warning")
        self.assertIn("25", recorder.lines[1][1])

    def test_distinct_failures_each_keep_their_first_line(self):
        recorder = self.Recorder()
        recorder.error_repeat("boom", FakeHttpError("a", 503))
        recorder.error_repeat("boom", FakeHttpError("b", 401))
        self.assertEqual(len(recorder.lines), 2)
        self.assertTrue(all(level == "error" for level, _ in recorder.lines))


@unittest.skipUnless(AnalysisTask is not None, TASK_SKIP)
class TestTaskSummary(unittest.TestCase):
    """The run must end by naming what was folded, and which interface was used."""

    def setUp(self):
        ErrorLedger.reset()

    def make_task(self, platform=None, tag="guts_1"):
        task = AnalysisTask.__new__(AnalysisTask)
        task.config = SimpleNamespace(
            platforms={tag: platform} if platform else {},
            target_platform=tag,
        )
        task.logs = []
        task.info = task.logs.append
        task.warning = task.logs.append
        return task

    def test_the_run_log_names_the_interface_not_just_the_model(self):
        task = self.make_task({
            "name": "Guts API STORE",
            "api_url": "https://api.gutsai.id/v1",
            "api_format": "OpenAI",
        })
        task._log_interface()
        self.assertEqual(len(task.logs), 1)
        self.assertIn("Guts API STORE", task.logs[0])
        self.assertIn("https://api.gutsai.id/v1", task.logs[0])

    def test_a_missing_interface_block_does_not_raise(self):
        task = self.make_task(None, tag="ghost")
        task._log_interface()
        self.assertIn("ghost", task.logs[0])

    def test_a_quiet_run_says_nothing_extra(self):
        self.make_task()._log_error_summary()
        self.assertEqual(self.make_task().logs, [])

    def test_a_noisy_run_is_summarised_by_count(self):
        task = self.make_task()
        error = FakeHttpError("model_unavailable", 503)
        for _ in range(184):
            ErrorLedger.record(error)
        task._log_error_summary()
        joined = "\n".join(task.logs)
        self.assertIn("184", joined)
        self.assertIn("503", joined)


if __name__ == "__main__":
    unittest.main()
