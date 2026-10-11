"""The run-start line that names the interface and what will be sent to it.

An extraction failed intermittently for a day: a custom gateway with think_switch on was
sent reasoning_effort=max, which the gateway sometimes refused with 503 model_unavailable.
Neither the config nor the log said so. The line this pins exists to make that visible
before a run rather than after an investigation.

The last test is the one that matters most: it is the bug this method shipped with. A
profile that puts its parameters inside extra_body was reported as sending nothing, which
is a false statement in the log - worse than no statement.
"""
import unittest

from types import SimpleNamespace

HEAVY_ROOTS = {
    "PyQt5", "qfluentwidgets", "rapidjson", "openai", "anthropic", "boto3", "botocore",
    "google", "httpx", "curl_cffi", "tiktoken", "rich", "chardet", "bs4", "mediapipe",
    "regex", "langcodes", "requests",
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


_task, TASK_SKIP = _load("ModuleFolders.Service.TaskExecutor.AnalysisTask")
AnalysisTask = getattr(_task, "AnalysisTask", None)

class FakeHttpError(Exception):
    """Stands in for an SDK error that carries an HTTP status."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code

CUSTOM_GATEWAY = {
    "name": "Guts API STORE",
    "group": "custom",
    "profile": None,
    "api_url": "https://api.gutsai.id/v1",
    "api_format": "OpenAI",
    "model": "gpt-6-luna",
    "extra_body": {},
}


def platform(**overrides):
    merged = dict(CUSTOM_GATEWAY)
    merged.update(overrides)
    return merged


class Recorder:
    def __init__(self):
        self.lines = []

    def info(self, message):
        self.lines.append(str(message))

    def warning(self, message):
        self.lines.append("WARNING: " + str(message))


@unittest.skipUnless(AnalysisTask is not None, TASK_SKIP)
class TestRequestShapingLog(unittest.TestCase):
    def log_for(self, platform_config):
        task = AnalysisTask.__new__(AnalysisTask)
        task.config = SimpleNamespace(platforms={"c1": platform_config}, target_platform="c1")
        recorder = Recorder()
        task.info = recorder.info
        task.warning = recorder.warning
        task._log_interface()
        task._log_request_shaping()
        return "\n".join(recorder.lines)

    def test_the_interface_is_named_not_just_the_model(self):
        text = self.log_for(platform(think_switch=False))
        self.assertIn("Guts API STORE", text)
        self.assertIn("https://api.gutsai.id/v1", text)

    def test_an_unrecognised_gateway_with_thinking_on_says_what_it_will_send(self):
        text = self.log_for(platform(think_switch=True, think_depth="max", temperature=1.05))
        self.assertIn("reasoning_effort=max", text)
        self.assertIn("未识别网关", text)
        # The field that actually broke the run must be visible, not implied.
        self.assertIn("temperature=1.05", text)

    def test_the_working_configuration_says_it_sends_nothing_extra(self):
        text = self.log_for(platform(think_switch=False, temperature=1.0))
        self.assertIn("不发送额外思考参数", text)
        self.assertNotIn("reasoning_effort", text)

    def test_parameters_carried_in_extra_body_are_not_reported_as_nothing(self):
        # The shipped bug: a profile that lands everything in extra_body was announced as
        # sending nothing extra, while the request carried it.
        text = self.log_for(platform(name="openrouterv2", profile="openrouter",
                                     api_url="https://openrouter.ai/api/v1",
                                     think_switch=True, think_depth="max", temperature=1.0))
        self.assertNotIn("不发送额外思考参数", text)
        self.assertIn("extra_body", text)

    def test_a_platform_with_no_config_does_not_raise(self):
        text = self.log_for(platform())
        self.assertIsInstance(text, str)


@unittest.skipUnless(AnalysisTask is not None, TASK_SKIP)
class TestReasoningEffortHint(unittest.TestCase):
    """The hint that turns a 503 investigation into one line.

    The failure shape is: an unrecognised gateway, thinking on, so a bare
    reasoning_effort goes out, and the gateway answers 503 model_unavailable. That reads
    like the model is offline, which is how a day went missing. The hint may only appear
    when all three hold - a hint that cries wolf is worse than none.
    """

    def setUp(self):
        from ModuleFolders.Log import ErrorLedger
        ErrorLedger.reset()
        for _ in range(3):
            ErrorLedger.record(FakeHttpError("model_unavailable", 503))

    def summary_for(self, platform_config):
        task = AnalysisTask.__new__(AnalysisTask)
        task.config = SimpleNamespace(platforms={"c1": platform_config}, target_platform="c1")
        lines = []
        task.info = lines.append
        task.warning = lines.append
        task._log_error_summary()
        return "\n".join(lines)

    def test_the_field_is_named_when_an_unknown_gateway_was_sent_it(self):
        text = self.summary_for(platform(think_switch=True, think_depth="high"))
        self.assertIn("reasoning_effort=high", text)
        self.assertIn("未识别网关", text)
        self.assertIn("关闭思考模式", text)

    def test_no_hint_when_nothing_extra_was_sent(self):
        text = self.summary_for(platform(think_switch=False, temperature=1.0))
        self.assertNotIn("reasoning_effort", text)

    def test_no_hint_for_a_gateway_whose_dialect_is_known(self):
        # OpenRouter carries its effort inside extra_body, so the OpenAI field is not the
        # suspect and the hint must stay quiet.
        text = self.summary_for(platform(name="openrouterv2", profile="openrouter",
                                         api_url="https://openrouter.ai/api/v1",
                                         think_switch=True, think_depth="high"))
        self.assertNotIn("reasoning_effort=", text)


class TestProfileChoices(unittest.TestCase):
    """Every dialect the code can resolve must be selectable in the interface settings.

    Adding a profile to the registry without offering it would leave users hand-editing
    config.json - which is the gap this selector exists to close.
    """

    def test_the_selector_offers_exactly_the_registry(self):
        from ModuleFolders.Infrastructure.LLMRequester.ThinkingProfiles import PROFILES, PROFILE_ORDER
        self.assertEqual(set(PROFILE_ORDER), set(PROFILES),
                         set(PROFILE_ORDER) ^ set(PROFILES))

    def test_the_selector_has_no_duplicates(self):
        from ModuleFolders.Infrastructure.LLMRequester.ThinkingProfiles import PROFILE_ORDER
        self.assertEqual(len(PROFILE_ORDER), len(set(PROFILE_ORDER)))


if __name__ == "__main__":
    unittest.main()
