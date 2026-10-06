"""LanguageChecker wiring: model path, switch, and fail-open fallback.

The decision logic itself is covered by test_language_verdict.py, which runs everywhere.
What is left to prove here is the glue: that the switch selects the model path, that a model
which is down or answers unreadably falls back to the local detector, and that a disabled
switch never calls the model at all.

This module needs the reader stack (mediapipe, chardet, bs4) because LanguageChecker imports
ReaderUtil at module level. Those are real declared dependencies, so the tests run in CI;
locally they SKIP rather than pretend. The skip is deliberately narrow: an ImportError from
any other module - especially our own - is re-raised, so a genuine import bug in this repo can
never masquerade as a skip.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

# SystemOneClient is the MODULE here (the tests patch .urlrequest.urlopen on it);
# Client is the class. Importing the class under the module's name breaks the patch.
from ModuleFolders.Infrastructure.DecisionEngine import DecisionEngine, Questions, SystemOneClient
from ModuleFolders.Infrastructure.DecisionEngine.SystemOneClient import SystemOneClient as Client
from tests.test_decision_engine import FakeResponse, sent_body


class TestHarnessIntegrity(unittest.TestCase):
    """Deliberately NOT skipped, so the harness is checked wherever the suite runs.

    The tests below skip when the reader stack is unavailable. That is honest, but it also
    means a mistake in the patch target would go unseen locally and only surface in CI - which
    is exactly what happened: SystemOneClient was imported as the class, so the patch raised
    AttributeError, and every test here skipped locally and errored in CI. These assertions
    run everywhere, so the target is verified even when the tests cannot run.
    """

    def test_the_urlopen_patch_target_is_the_module_not_the_class(self):
        import types
        self.assertIsInstance(SystemOneClient, types.ModuleType,
                              "patch .urlrequest on the module, not the class")
        self.assertTrue(hasattr(SystemOneClient, "urlrequest"))
        self.assertTrue(callable(SystemOneClient.urlrequest.urlopen))

    def test_the_client_factory_name_is_the_class(self):
        self.assertIsInstance(Client, type)
        self.assertTrue(hasattr(Client, "evaluate"))

# Declared dependencies that are simply not installed in every environment.
HEAVY_ROOTS = {
    "mediapipe", "PyQt5", "qfluentwidgets", "chardet", "bs4", "rich", "tiktoken",
    "langdetect", "openai", "anthropic", "boto3", "botocore", "google", "httpx",
    "curl_cffi", "requests", "regex", "langcodes",
}


def _load():
    try:
        import ModuleFolders.Service.TranslationChecker.LanguageChecker as module
        return module, module.LanguageChecker, ""
    except ImportError as error:
        missing = getattr(error, "name", "") or ""
        if missing.split(".")[0] not in HEAVY_ROOTS:
            raise
        return None, None, "not installed here: {}".format(missing)


LanguageCheckerModule, LanguageChecker, SKIP_REASON = _load()


def make_checker(engine, use_model=True):
    """A real LanguageChecker with only the state the decision path touches."""
    checker = LanguageChecker.__new__(LanguageChecker)
    checker._decision_engine = engine
    checker._use_decision_model = use_model
    # ja target from a zh source is the configuration the local rule cannot handle.
    checker._last_target_language_name = "japanese"
    checker._last_target_language_code = "ja"
    checker._last_source_language_name = "chinese_simplified"
    checker._last_source_language_code = "zh"
    checker.warnings = []
    checker.warning = checker.warnings.append
    return checker


def make_engine():
    return DecisionEngine.DecisionEngine(Client(api_key="k", model="jev-latest"))


def item(source, translated):
    return SimpleNamespace(source_text=source, translated_text=translated)


def reply_with(probabilities):
    def fake_urlopen(request, timeout=None):
        body = sent_body(request)
        answers = {}
        for qid in body["questions"]:
            index = int(qid[1:].split("_")[0])
            in_target, still_source = probabilities[index]
            answers[qid] = {"type": "noul", "noul": in_target if qid.endswith("_lang") else still_source}
        return FakeResponse({"answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}})
    return fake_urlopen


@unittest.skipUnless(LanguageChecker is not None, SKIP_REASON)
class TestWiring(unittest.TestCase):
    LOCAL = [(["ja"], 0.99)]

    def _run(self, checker, items):
        with mock.patch.object(LanguageCheckerModule.ReaderUtil, "detect_language_with_mediapipe",
                               return_value=self.LOCAL) as local:
            result = checker._run_detection(items, "translated_text")
        return result, local

    def test_a_model_verdict_is_used_and_the_local_detector_is_never_called(self):
        checker = make_checker(make_engine())
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen",
                               side_effect=reply_with({0: (0.95, 0.02)})):
            result, local = self._run(checker, [item("a", "A")])
        self.assertEqual(result, [(["ja"], 0.95)])
        local.assert_not_called()

    def test_an_untranslated_line_survives_the_whole_way_out(self):
        checker = make_checker(make_engine())
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen",
                               side_effect=reply_with({0: (0.03, 0.97)})):
            result, _ = self._run(checker, [item("SRC", "SRC")])
        self.assertEqual(result, [(["zh"], 0.97)])

    def test_the_request_carries_both_language_labels_and_the_line_text(self):
        checker = make_checker(make_engine())
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured.update(sent_body(request))
            return reply_with({0: (0.9, 0.1)})(request)

        with mock.patch.object(SystemOneClient.urlrequest, "urlopen", side_effect=fake_urlopen):
            self._run(checker, [item("SRC", "DST")])

        self.assertEqual(captured["state"]["items"]["0"], {"source": "SRC", "translated": "DST"})
        self.assertIn("japanese", captured["state"]["target_language"])
        self.assertIn("chinese_simplified", captured["state"]["source_language"])

    def test_a_batch_larger_than_the_question_cap_is_split_but_fully_answered(self):
        checker = make_checker(make_engine())
        items = [item("s%d" % i, "t%d" % i) for i in range(40)]
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(len(sent_body(request)["questions"]))
            return reply_with({i: (0.9, 0.05) for i in range(40)})(request)

        with mock.patch.object(SystemOneClient.urlrequest, "urlopen", side_effect=fake_urlopen):
            result, local = self._run(checker, items)

        self.assertEqual(len(result), 40)
        self.assertGreater(len(calls), 1)
        self.assertTrue(all(count <= Questions.MAX_QUESTIONS for count in calls))
        local.assert_not_called()

    def test_an_unreadable_answer_falls_back_to_the_local_detector(self):
        checker = make_checker(make_engine())
        items = [item("a", "A"), item("b", "B")]

        def partial(request, timeout=None):
            return FakeResponse({"answers": {"i0_lang": {"type": "noul", "noul": 0.9},
                                             "i0_residual": {"type": "noul", "noul": 0.1}}})

        with mock.patch.object(SystemOneClient.urlrequest, "urlopen", side_effect=partial):
            result, local = self._run(checker, items)
        self.assertEqual(result, self.LOCAL)
        local.assert_called_once()
        self.assertEqual(len(checker.warnings), 1)

    def test_a_provider_outage_falls_back_instead_of_flagging_every_line(self):
        from urllib.error import URLError
        checker = make_checker(make_engine())
        items = [item("s%d" % i, "t%d" % i) for i in range(50)]
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen", side_effect=URLError("down")):
            result, local = self._run(checker, items)
        self.assertEqual(result, self.LOCAL)
        local.assert_called_once()

    def test_the_switch_off_means_pure_local_behaviour(self):
        checker = make_checker(make_engine(), use_model=False)
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen") as post:
            result, local = self._run(checker, [item("a", "A")])
        self.assertEqual(result, self.LOCAL)
        local.assert_called_once()
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()