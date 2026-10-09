"""The run log must say whether the decision model ran, and why it failed.

Three silences used to look identical: the switch is off, no decision interface is
configured, and an interface is configured whose every call fails. The first is expected,
the second is a setup mistake, and the third is an outage with money attached - so the log
has to tell them apart. A failure also has to carry enough to act on months later: which
kind of failure, the HTTP status when there was one, and the provider's own words.

The engine-side contract runs everywhere. The two call sites import Qt-bound modules, so
that half skips locally and runs in CI, with the skip narrowed to known heavy dependencies
so a real import bug cannot hide as a skip.
"""
import json
import unittest
from types import SimpleNamespace
from unittest import mock
from urllib.error import HTTPError, URLError

from ModuleFolders.Infrastructure.DecisionEngine import DecisionEngine, Questions, SystemOneClient
from ModuleFolders.Infrastructure.DecisionEngine.SystemOneClient import DecisionError

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "openai/gpt-6-luna-decisions"

HEAVY_ROOTS = {
    "PyQt5", "qfluentwidgets", "rapidjson", "openai", "anthropic", "boto3", "botocore",
    "google", "httpx", "curl_cffi", "tiktoken", "rich", "chardet", "bs4", "mediapipe",
    "regex", "langcodes",
}


def _load(name):
    import importlib
    try:
        return importlib.import_module(name), ""
    except ImportError as error:
        missing = getattr(error, "name", "") or ""
        if missing.split(".")[0] not in HEAVY_ROOTS:
            raise
        return None, "not installed here: {}".format(missing)


_analysis, ANALYSIS_SKIP = _load("ModuleFolders.Service.TaskExecutor.AnalysisTask")
_checker, CHECKER_SKIP = _load("ModuleFolders.Service.TranslationChecker.LanguageChecker")
AnalysisTask = getattr(_analysis, "AnalysisTask", None)
LanguageChecker = getattr(_checker, "LanguageChecker", None)


class FakeResponse:
    """Minimal stand-in for the object urlopen yields (a context manager)."""

    def __init__(self, payload=None, raw=None):
        self._raw = raw if raw is not None else json.dumps(payload).encode("utf-8")

    def read(self):
        return self._raw

    def close(self):
        """HTTPError is file-like and closes its fp on dealloc."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def patch(return_value=None, side_effect=None):
    return mock.patch.object(SystemOneClient.urlrequest, "urlopen",
                             return_value=return_value, side_effect=side_effect)


def client(**overrides):
    options = dict(api_key="secret", model=MODEL, shape=SystemOneClient.DECISIONS,
                   base_url="https://openrouter.ai/api")
    options.update(overrides)
    return SystemOneClient.SystemOneClient(**options)


def make_engine(**overrides):
    """An engine over a stub client: no transport, but the real accounting code."""
    stub = mock.Mock()
    stub.model = overrides.get("model", MODEL)
    stub.shape = overrides.get("shape", SystemOneClient.DECISIONS)
    stub.endpoint = overrides.get("endpoint", URL)
    stub.evaluate.side_effect = overrides.get("side_effect", None)
    return DecisionEngine.DecisionEngine(stub)


def fail(engine, count=1, error=None, purpose="术语一致性巡检"):
    """Drive a real ask() so the failures come through the production path."""
    engine.client.evaluate.side_effect = error or DecisionError("boom", kind="http", status=500)
    for _ in range(count):
        engine.ask({"items": {}}, {"q0": Questions.noul("ok?")}, purpose=purpose)


class TestErrorDetail(unittest.TestCase):
    """A failure has to name its own kind; "it broke" is not a diagnosis."""

    def test_an_http_failure_carries_the_status_and_the_providers_words(self):
        error = HTTPError(URL, 402, "Payment Required", {},
                          FakeResponse(raw=b'{"error":{"message":"Insufficient credits"}}'))
        with patch(side_effect=error):
            with self.assertRaises(DecisionError) as caught:
                client().evaluate("state", {"q0": Questions.noul("ok?")})
        self.assertEqual(caught.exception.kind, "http")
        self.assertEqual(caught.exception.status, 402)
        self.assertIn("Insufficient credits", caught.exception.detail)

    def test_a_transport_failure_is_distinguishable_from_an_http_one(self):
        with patch(side_effect=URLError("connection refused")):
            with self.assertRaises(DecisionError) as caught:
                client().evaluate("state", {"q0": Questions.noul("ok?")})
        self.assertEqual(caught.exception.kind, "transport")
        self.assertIsNone(caught.exception.status)

    def test_an_unreadable_body_is_a_shape_failure(self):
        with patch(return_value=FakeResponse(raw=b"not json")):
            with self.assertRaises(DecisionError) as caught:
                client().evaluate("state", {"q0": Questions.noul("ok?")})
        self.assertEqual(caught.exception.kind, "shape")

    def test_a_missing_answers_map_is_a_shape_failure(self):
        with patch(return_value=FakeResponse({"model": MODEL})):
            with self.assertRaises(DecisionError) as caught:
                client().evaluate("state", {"q0": Questions.noul("ok?")})
        self.assertEqual(caught.exception.kind, "shape")

    def test_a_misconfiguration_is_a_config_failure(self):
        with self.assertRaises(DecisionError) as caught:
            client(api_key="").evaluate("state", {"q0": Questions.noul("ok?")})
        self.assertEqual(caught.exception.kind, "config")

    def test_the_message_is_unchanged_for_callers_that_only_print_it(self):
        with patch(side_effect=URLError("down")):
            with self.assertRaises(DecisionError) as caught:
                client().evaluate("state", {"q0": Questions.noul("ok?")})
        self.assertIn("decision request failed", str(caught.exception))


class TestFailureRecords(unittest.TestCase):
    def test_a_failure_is_recorded_with_its_kind_status_and_purpose(self):
        engine = make_engine()
        fail(engine, error=DecisionError("boom", kind="http", status=429, detail="slow down"))
        self.assertEqual(engine.failures, 1)
        self.assertEqual(len(engine.failure_detail), 1)
        record = engine.failure_detail[0]
        self.assertEqual(record["kind"], "http")
        self.assertEqual(record["status"], 429)
        self.assertEqual(record["purpose"], "术语一致性巡检")
        self.assertIn("boom", record["error"])

    def test_the_last_error_is_still_kept_for_existing_readers(self):
        engine = make_engine()
        fail(engine, error=DecisionError("boom", kind="transport"))
        self.assertIn("boom", engine.last_error)
        self.assertIn("last_error", engine.summary())

    def test_failures_are_recorded_only_up_to_the_cap_and_counted_beyond_it(self):
        engine = make_engine()
        cap = DecisionEngine.MAX_FAILURE_DETAIL
        fail(engine, cap + 5, error=DecisionError("boom", kind="http", status=500))
        self.assertEqual(engine.failures, cap + 5)
        self.assertEqual(len(engine.failure_detail), cap)
        lines = engine.failure_lines()
        self.assertEqual(len(lines), cap + 1)
        self.assertIn("5", lines[-1])

    def test_a_successful_call_records_nothing(self):
        engine = make_engine()
        engine.client.evaluate.side_effect = None
        engine.client.evaluate.return_value = {"answers": {"q0": {"type": "noul", "noul": 0.9}}}
        engine.ask({"items": {}}, {"q0": Questions.noul("ok?")})
        self.assertEqual(engine.failures, 0)
        self.assertEqual(engine.failure_detail, [])


class TestLogLines(unittest.TestCase):
    """The line has to answer: which model, where, how much, and what went wrong."""

    def test_describe_names_the_model_the_endpoint_and_the_counters(self):
        engine = make_engine()
        fail(engine, error=DecisionError("boom", kind="transport"))
        text = engine.describe()
        self.assertIn(MODEL, text)
        self.assertIn(URL, text)
        self.assertIn(SystemOneClient.DECISIONS, text)
        self.assertIn("调用 1 次", text)
        self.assertIn("失败 1 次", text)

    def test_describe_survives_a_client_that_has_nothing_to_say(self):
        engine = make_engine(model="", endpoint="", shape="")
        self.assertIn("(未填写模型)", engine.describe())

    def test_a_failure_line_carries_the_kind_and_the_status(self):
        engine = make_engine()
        fail(engine, error=DecisionError("Insufficient credits", kind="http", status=402))
        line = engine.failure_lines()[0]
        self.assertIn("http/HTTP 402", line)
        self.assertIn("Insufficient credits", line)
        self.assertIn("术语一致性巡检", line)

    def test_a_status_free_failure_says_so_without_inventing_one(self):
        engine = make_engine()
        fail(engine, error=DecisionError("down", kind="transport"))
        self.assertIn("transport", engine.failure_lines()[0])
        self.assertNotIn("HTTP", engine.failure_lines()[0])

    def test_summary_gains_the_identity_keys_without_losing_the_old_ones(self):
        engine = make_engine()
        summary = engine.summary()
        for key in ("calls", "failures", "questions", "input_tokens", "output_tokens",
                    "cost", "last_error"):
            self.assertIn(key, summary)
        self.assertEqual(summary["model"], MODEL)
        self.assertEqual(summary["endpoint"], URL)
        self.assertEqual(summary["shape"], SystemOneClient.DECISIONS)


@unittest.skipUnless(AnalysisTask is not None, ANALYSIS_SKIP)
class TestAnalysisTaskLogging(unittest.TestCase):
    """The extraction run must say whether the decision layer took part."""

    def make_task(self, engine, sweep_switch=True):
        task = AnalysisTask.__new__(AnalysisTask)
        task.config = SimpleNamespace(extract_consistency_sweep_switch=sweep_switch)
        task._decision_engine = engine
        task._decision_engine_resolved = True
        task.logs = []
        task.info = task.logs.append
        task.warning = task.logs.append
        return task

    def test_an_enabled_layer_names_the_model_and_the_endpoint(self):
        task = self.make_task(make_engine())
        task._log_decision_layer()
        self.assertEqual(len(task.logs), 1)
        self.assertIn(MODEL, task.logs[0])
        self.assertIn(URL, task.logs[0])

    def test_a_disabled_switch_says_the_layer_was_not_used(self):
        task = self.make_task(make_engine(), sweep_switch=False)
        task._log_decision_layer()
        self.assertEqual(len(task.logs), 1)
        self.assertIn("未使用", task.logs[0])

    def test_no_engine_is_not_reported_twice(self):
        """The unconfigured case is explained by _consistency_engine, once."""
        task = self.make_task(None)
        task._log_decision_layer()
        self.assertEqual(task.logs, [])

    def test_an_untouched_layer_stays_quiet_at_the_end(self):
        task = self.make_task(make_engine())
        task._log_decision_summary()
        self.assertEqual(task.logs, [])

    def test_a_clean_run_reports_usage(self):
        engine = make_engine()
        engine.client.evaluate.side_effect = None
        engine.client.evaluate.return_value = {"answers": {}, "usage": {"input_tokens": 12}}
        engine.ask({"items": {}}, {"q0": Questions.noul("ok?")})
        task = self.make_task(engine)
        task._log_decision_summary()
        self.assertEqual(len(task.logs), 1)
        self.assertIn("调用 1 次", task.logs[0])

    def test_a_failing_run_reports_each_failure_with_its_status(self):
        engine = make_engine()
        fail(engine, error=DecisionError("Insufficient credits", kind="http", status=402))
        task = self.make_task(engine)
        task._log_decision_summary()
        self.assertIn("失败", task.logs[0])
        self.assertTrue(any("http/HTTP 402" in line for line in task.logs[1:]))


@unittest.skipUnless(LanguageChecker is not None, CHECKER_SKIP)
class TestLanguageCheckerLogging(unittest.TestCase):
    """The language check must tell its three silences apart."""

    def make_checker(self, engine, use=True, wants=None):
        checker = LanguageChecker.__new__(LanguageChecker)
        checker._decision_engine = engine
        # Mirrors the real invariant from run_check: without an engine the layer can never
        # be in use, and "wanted" records what the user asked for whatever the outcome.
        checker._use_decision_model = use and engine is not None
        checker._wants_decision_model = bool(use) if wants is None else wants
        checker.logs = []
        checker.info = checker.logs.append
        checker.warning = checker.logs.append
        return checker

    def test_an_unconfigured_checker_says_so_once(self):
        checker = self.make_checker(None, use=False)
        checker._log_decision_layer()
        self.assertEqual(len(checker.logs), 1)
        self.assertIn("未使用", checker.logs[0])
        self.assertIn("未配置", checker.logs[0])

    def test_wanted_but_unconfigured_is_not_said_twice(self):
        """run_check already warns in that case; this must not repeat the same sentence."""
        checker = self.make_checker(None, use=False, wants=True)
        checker._log_decision_layer()
        self.assertEqual(checker.logs, [])

    def test_a_configured_but_switched_off_checker_says_so_differently(self):
        checker = self.make_checker(make_engine(), use=False)
        checker._log_decision_layer()
        self.assertIn("未使用", checker.logs[0])
        self.assertIn("开关", checker.logs[0])

    def test_an_active_checker_reports_usage(self):
        engine = make_engine()
        engine.client.evaluate.side_effect = None
        engine.client.evaluate.return_value = {"answers": {}, "usage": {"input_tokens": 5}}
        engine.ask({"items": {}}, {"q0": Questions.noul("ok?")})
        checker = self.make_checker(engine)
        checker._log_decision_layer()
        self.assertIn(MODEL, checker.logs[0])


@unittest.skipUnless(LanguageChecker is not None, CHECKER_SKIP)
class TestCheckerResolution(unittest.TestCase):
    """LanguageChecker resolves its engine in __init__ - the other site of the same bug.

    from_config is a classmethod on DecisionEngine the class, while the module carries the
    same name, so the bare module lookup raised AttributeError the moment the checker was
    constructed. Nothing covered it: every other test here builds a checker with __new__,
    which skips __init__ entirely.
    """

    CONFIG = {
        "platforms": {
            "luna_decisions_482913": {
                "group": "decision",
                "api_format": "decisions",
                "api_url": "https://openrouter.ai/api",
                "api_key": "k",
                "model": "openai/gpt-6-luna-decisions",
            }
        }
    }

    def make_checker(self, config):
        with mock.patch.object(LanguageChecker, "load_config", return_value=config):
            return LanguageChecker(cache_manager=None)

    def test_construction_resolves_the_decision_engine(self):
        checker = self.make_checker(dict(self.CONFIG))
        self.assertIsNotNone(checker._decision_engine)
        self.assertEqual(checker._decision_engine.endpoint,
                         "https://openrouter.ai/api/alpha/decisions")

    def test_construction_without_a_decision_platform_is_fine(self):
        checker = self.make_checker({"platforms": {}})
        self.assertIsNone(checker._decision_engine)
        self.assertFalse(checker._use_decision_model)


if __name__ == "__main__":
    unittest.main()
