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


if __name__ == "__main__":
    unittest.main()
