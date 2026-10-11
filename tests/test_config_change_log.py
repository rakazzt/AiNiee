"""Recording what a settings save actually changed.

A user turned think_switch off and set temperature to 1.00 on one interface. An extraction
that had failed for a day started succeeding - and the log contained no record of either
change. Finding out meant diffing two installations' config.json by hand.

save_config is the single place that sees both the previous config and the new one, so the
diff lives there and the rule lives here, free of Qt, of the config object and of the
logger. Two properties matter beyond "it reports changes": a key must never reach the log,
and neither must a value long enough to drown it.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

from ModuleFolders.Config import ConfigChangeLog

HEAVY_ROOTS = {"rapidjson", "rich", "PyQt5", "qfluentwidgets", "requests", "openai"}


def _load(name):
    import importlib
    try:
        return importlib.import_module(name), ""
    except ImportError as error:
        missing = getattr(error, "name", "") or ""
        if missing.split(".")[0] not in HEAVY_ROOTS:
            raise
        return None, "not installed here: {}".format(missing)


_config, CONFIG_SKIP = _load("ModuleFolders.Config.Config")
ConfigMixin = getattr(_config, "ConfigMixin", None)


def platform(name, **overrides):
    base = {
        "name": name,
        "group": "custom",
        "api_url": "https://api.example.test/v1",
        "api_format": "OpenAI",
        "model": "some-model",
        "think_switch": False,
        "think_depth": "low",
        "temperature": 1.0,
        "extra_body": {},
        "api_key": "sk-original",
    }
    base.update(overrides)
    return base


class TestDescribeChanges(unittest.TestCase):
    def test_nothing_changed_reports_nothing(self):
        same = {"platforms": {"p1": platform("Guts")}}
        self.assertEqual(ConfigChangeLog.describe(same, json.loads(json.dumps(same))), [])

    def test_the_exact_change_that_fixed_the_outage_is_named(self):
        old = {"platforms": {"p1": platform("Guts API STORE", think_switch=True,
                                            think_depth="max", temperature=1.05)}}
        new = {"platforms": {"p1": platform("Guts API STORE", think_switch=False,
                                            think_depth="max", temperature=1.0)}}
        lines = ConfigChangeLog.describe(old, new)
        joined = "\n".join(lines)
        self.assertIn("think_switch", joined)
        self.assertIn("temperature", joined)
        self.assertIn("Guts API STORE", joined)
        # Booleans and numbers must read as values, not as Python reprs.
        self.assertIn("开 → 关", joined)
        self.assertIn("1.05 → 1.0", joined)

    def test_a_key_is_reported_as_present_or_absent_never_as_text(self):
        old = {"platforms": {"p1": platform("Guts", api_key="sk-secret-old")}}
        new = {"platforms": {"p1": platform("Guts", api_key="sk-secret-new")}}
        joined = "\n".join(ConfigChangeLog.describe(old, new))
        self.assertNotIn("sk-secret-old", joined)
        self.assertNotIn("sk-secret-new", joined)
        self.assertIn("api_key", joined)

    def test_rebinding_a_role_says_which_interface_each_role_uses(self):
        old = {"api_settings": {"extract": "p1"}, "platforms": {"p1": platform("Old One")}}
        new = {"api_settings": {"extract": "p2"}, "platforms": {"p2": platform("New One")}}
        joined = "\n".join(ConfigChangeLog.describe(old, new))
        self.assertIn("extract", joined)
        self.assertIn("Old One", joined)
        self.assertIn("New One", joined)

    def test_adding_and_removing_an_interface_is_reported(self):
        old = {"platforms": {"p1": platform("Keeper"), "p2": platform("Doomed")}}
        new = {"platforms": {"p1": platform("Keeper"), "p3": platform("Fresh")}}
        joined = "\n".join(ConfigChangeLog.describe(old, new))
        self.assertIn("Fresh", joined)
        self.assertIn("Doomed", joined)

    def test_a_long_prompt_change_is_paraphrased_not_dumped(self):
        old = {"platforms": {"p1": platform("Guts")}, "extract_common_prompt_user": "a" * 5000}
        new = {"platforms": {"p1": platform("Guts")}, "extract_common_prompt_user": "b" * 5000}
        lines = ConfigChangeLog.describe(old, new)
        for line in lines:
            self.assertLess(len(line), 400, line)

    def test_many_changes_are_counted_not_listed_without_end(self):
        old = {"platforms": {}}
        new = {"platforms": {"p{0}".format(i): platform("N{0}".format(i)) for i in range(40)}}
        lines = ConfigChangeLog.describe(old, new)
        self.assertLessEqual(len(lines), ConfigChangeLog.MAX_REPORTED_CHANGES + 1)
        self.assertIn("未列出", lines[-1])

    def test_junk_input_does_not_raise(self):
        for old, new in ((None, None), ({}, {"platforms": None}), ({"platforms": []}, {})):
            self.assertIsInstance(ConfigChangeLog.describe(old, new), list)


@unittest.skipUnless(ConfigMixin is not None, CONFIG_SKIP)
class TestSaveConfigRecords(unittest.TestCase):
    """save_config is the one writer, so it is the one place that can keep this record."""

    # The base must exist at class-definition time even when the module is skipped here.
    class Saver(ConfigMixin if ConfigMixin is not None else object):
        def __init__(self):
            self.logged = []

        def info(self, message):
            self.logged.append(message)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "config.json")
        # save_config re-derives CONFIG_PATH from config_path() on every call, so patching
        # the attribute alone would let the test write to the real user config.
        self.patches = [
            mock.patch.object(_config, "config_path", lambda: self.path),
            mock.patch.object(_config, "migrate_config_if_needed", lambda: None),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.tmp.cleanup()

    def write(self, config):
        with open(self.path, "w", encoding="utf-8") as writer:
            json.dump(config, writer)

    def test_a_save_that_changes_a_setting_says_so(self):
        self.write({"platforms": {"p1": platform("Guts", think_switch=True)}})
        saver = self.Saver()
        saver.save_config({"platforms": {"p1": platform("Guts", think_switch=False)}})
        joined = "\n".join(saver.logged)
        self.assertIn("配置已更改", joined)
        self.assertIn("think_switch", joined)
        # The change is on disk as well as in the log.
        with open(self.path, encoding="utf-8") as reader:
            self.assertFalse(json.load(reader)["platforms"]["p1"]["think_switch"])

    def test_a_save_with_no_change_says_nothing(self):
        config = {"platforms": {"p1": platform("Guts")}}
        self.write(config)
        saver = self.Saver()
        saver.save_config(json.loads(json.dumps(config)))
        self.assertEqual(saver.logged, [])

    def test_a_broken_logger_cannot_break_saving(self):
        self.write({"platforms": {"p1": platform("Guts", temperature=1.05)}})

        class Exploding(ConfigMixin if ConfigMixin is not None else object):
            def info(self, message):
                raise RuntimeError("logger is down")

        Exploding().save_config({"platforms": {"p1": platform("Guts", temperature=1.0)}})
        with open(self.path, encoding="utf-8") as reader:
            self.assertEqual(json.load(reader)["platforms"]["p1"]["temperature"], 1.0)

    def test_a_config_object_without_a_logger_still_saves(self):
        self.write({"platforms": {"p1": platform("Guts", temperature=1.05)}})
        ConfigMixin().save_config({"platforms": {"p1": platform("Guts", temperature=1.0)}})
        with open(self.path, encoding="utf-8") as reader:
            self.assertEqual(json.load(reader)["platforms"]["p1"]["temperature"], 1.0)


if __name__ == "__main__":
    unittest.main()
