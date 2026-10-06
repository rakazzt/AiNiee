"""Decision settings, and the two places they must actually take effect.

Two failure modes are worth more than the rest here:

1. A setting that is shown in the UI but changes nothing. Each switch is driven through the
   sweep and asserted against the requests it does or does not make.
2. A built-in prompt that can be selected but never used. build_system() only honoured USER
   prompts, so picking the decision variant would have changed the card and nothing else.
"""
import json
import unittest
from pathlib import Path
from unittest import mock

from ModuleFolders.Domain.PromptBuilder.PromptBuilderEnum import PromptBuilderEnum
from ModuleFolders.Domain.PromptBuilder.PromptBuilderExtraction import PromptBuilderExtraction
from ModuleFolders.Infrastructure.DecisionEngine import DecisionSettings
from ModuleFolders.Infrastructure.DecisionEngine import SystemOneClient
from ModuleFolders.Infrastructure.DecisionEngine.ConsistencySweep import ConsistencySweep
from tests.test_consistency_sweep import (ARTHUR, FENGSHEN, FENGSHEN_ZHAN, make_engine, replies,
                                          run_sources, term)

BASIC = PromptBuilderExtraction.BASIC
JUDGMENT = PromptBuilderExtraction.JUDGMENT


class TestNormalize(unittest.TestCase):
    def test_every_key_gets_a_value_even_from_nothing(self):
        self.assertEqual(DecisionSettings.normalize(None), DecisionSettings.DEFAULT_SETTINGS)
        self.assertEqual(DecisionSettings.normalize("nonsense"), DecisionSettings.DEFAULT_SETTINGS)
        self.assertEqual(DecisionSettings.normalize({}), DecisionSettings.DEFAULT_SETTINGS)

    def test_unknown_keys_are_dropped_not_carried(self):
        resolved = DecisionSettings.normalize({"drop_everything": True, "threshold": 0.7})
        self.assertNotIn("drop_everything", resolved)
        self.assertEqual(set(resolved), set(DecisionSettings.DEFAULT_SETTINGS))

    def test_booleans_accept_the_shapes_a_config_file_may_hold(self):
        for raw, expected in (("false", False), ("0", False), ("no", False), (0, False),
                              ("true", True), (1, True), ("yes", True), (True, True)):
            with self.subTest(raw=raw):
                self.assertIs(DecisionSettings.normalize({"relation_switch": raw})["relation_switch"],
                              expected)
        # Anything unrecognised falls back to the default rather than guessing.
        self.assertIs(DecisionSettings.normalize({"relation_switch": "maybe"})["relation_switch"], True)
        self.assertIs(DecisionSettings.normalize({"relation_switch": None})["relation_switch"], True)

    def test_threshold_and_cap_are_clamped_into_range(self):
        low, high = DecisionSettings.THRESHOLD_RANGE
        self.assertEqual(DecisionSettings.normalize({"threshold": -3})["threshold"], low)
        self.assertEqual(DecisionSettings.normalize({"threshold": 99})["threshold"], high)
        self.assertEqual(DecisionSettings.normalize({"threshold": "0.8"})["threshold"], 0.8)
        self.assertEqual(DecisionSettings.normalize({"threshold": "abc"})["threshold"],
                         DecisionSettings.DEFAULT_SETTINGS["threshold"])
        cap_low, cap_high = DecisionSettings.MAX_PAIRS_RANGE
        self.assertEqual(DecisionSettings.normalize({"max_pairs": 0})["max_pairs"], cap_low)
        self.assertEqual(DecisionSettings.normalize({"max_pairs": 10 ** 9})["max_pairs"], cap_high)


class TestSelection(unittest.TestCase):
    def test_without_a_selection_the_built_in_default_applies(self):
        self.assertEqual(DecisionSettings.get_selected({}), DecisionSettings.DEFAULT_SETTINGS)
        self.assertEqual(DecisionSettings.get_selected(None), DecisionSettings.DEFAULT_SETTINGS)

    def test_a_selected_custom_entry_is_used(self):
        config = {
            DecisionSettings.USER_KEY: [
                {"id": "abc", "name": "only report",
                 "settings": {"drop_generic": False, "threshold": 0.8}},
            ],
            DecisionSettings.SELECTION_KEY: {"last_selected_id": "abc"},
        }
        resolved = DecisionSettings.get_selected(config)
        self.assertFalse(resolved["drop_generic"])
        self.assertEqual(resolved["threshold"], 0.8)

    def test_a_selection_pointing_at_a_deleted_entry_falls_back(self):
        config = {DecisionSettings.SELECTION_KEY: {"last_selected_id": "gone",
                                                   "settings": {"drop_generic": False}}}
        self.assertEqual(DecisionSettings.get_selected(config), DecisionSettings.DEFAULT_SETTINGS)

    def test_malformed_user_entries_are_ignored_rather_than_raising(self):
        config = {DecisionSettings.USER_KEY: [None, "x", {}, {"id": "", "name": "n"},
                                              {"id": "ok", "name": "n", "settings": None}]}
        entries = DecisionSettings.get_user_settings(config)
        self.assertEqual([e["id"] for e in entries], ["ok"])
        self.assertEqual(entries[0]["settings"], DecisionSettings.DEFAULT_SETTINGS)

    def test_summarize_is_readable_and_survives_junk(self):
        text = DecisionSettings.summarize({"threshold": 0.75, "relation_switch": False})
        self.assertIn("0.75", text)
        self.assertIsInstance(DecisionSettings.summarize(None), str)


class TestSweepHonoursSettings(unittest.TestCase):
    """Every switch is driven end to end: no setting may be decorative."""

    def sweep(self, settings, terms=None, **kwargs):
        calls = []
        sweep = ConsistencySweep(make_engine(), settings=settings)
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen",
                               side_effect=replies(calls=calls, **kwargs)):
            return sweep.sweep_sources(terms or [term(FENGSHEN, "character"), term(FENGSHEN_ZHAN)]), calls

    def asked(self, calls):
        return [qid[0] for body in calls for qid in body["questions"]]

    def test_relation_switch_off_asks_nothing_about_pairs(self):
        outcome, calls = self.sweep({"relation_switch": False})
        self.assertNotIn("r", self.asked(calls))
        self.assertEqual(outcome.keep_separate, set())
        self.assertEqual(outcome.pairs_total, 1, "the pair is still found, just not judged")

    def test_generic_switch_off_asks_nothing_about_generic_terms(self):
        outcome, calls = self.sweep({"generic_switch": False})
        self.assertNotIn("g", self.asked(calls))
        self.assertEqual(outcome.generic, [])

    def test_both_switches_off_makes_no_request_at_all(self):
        outcome, calls = self.sweep({"generic_switch": False, "relation_switch": False})
        self.assertEqual(calls, [])
        self.assertEqual(outcome.keep_separate, set())

    def test_a_single_character_is_kept_and_judged_when_dropping_is_off(self):
        outcome, calls = self.sweep(
            {"drop_single_character": False, "generic_switch": True},
            terms=[term("\u5251"), term(FENGSHEN)],
            generic={"\u5251": 0.99},
        )
        self.assertEqual(outcome.dropped, ["\u5251"], "the generic verdict still removes it")
        self.assertIn("g", self.asked(calls), "and it was actually put to the model")

    def test_a_single_character_is_removed_without_a_request_when_dropping_is_on(self):
        outcome, calls = self.sweep(
            {"drop_single_character": True, "generic_switch": True},
            terms=[term("\u5251"), term(FENGSHEN)],
        )
        self.assertEqual(outcome.single_character, ["\u5251"])
        self.assertEqual(outcome.dropped, ["\u5251"])
        for body in calls:
            for item in body["state"]["items"].values():
                self.assertNotEqual(item.get("source"), "\u5251")

    def test_a_generic_term_is_reported_but_kept_when_dropping_is_off(self):
        outcome, _ = self.sweep(
            {"drop_generic": False},
            terms=[term("\u5927\u4eba"), term(FENGSHEN)],
            generic={"\u5927\u4eba": 0.95, FENGSHEN: 0.01},
        )
        self.assertEqual(outcome.generic, ["\u5927\u4eba"], "still reported")
        self.assertEqual(outcome.dropped, [], "but not removed")

    def test_a_raised_threshold_turns_a_borderline_verdict_into_unknown(self):
        decisive, _ = self.sweep({}, relations={(FENGSHEN, FENGSHEN_ZHAN): "subordinate"})
        self.assertIn((FENGSHEN, FENGSHEN_ZHAN), decisive.keep_separate)
        strict, _ = self.sweep({"threshold": 0.95},
                               relations={(FENGSHEN, FENGSHEN_ZHAN): "subordinate"})
        self.assertEqual(strict.keep_separate, set(), "0.90 leaves it undecided")

    def test_the_pair_cap_comes_from_the_settings(self):
        sources = [ARTHUR, ARTHUR + "\u4e4b\u5251", ARTHUR + "\u4e4b\u5251\u4e4b\u5f71"]
        outcome, _ = self.sweep({"max_pairs": 1}, terms=[term(s) for s in sources])
        self.assertEqual(outcome.pairs_total, 1)
        self.assertGreater(outcome.pairs_dropped, 0)

    def test_consistency_switch_off_makes_no_translation_request(self):
        sweep = ConsistencySweep(make_engine(), settings={"consistency_switch": False})
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen") as post:
            outcome = sweep.sweep_translations(
                [{"source": FENGSHEN, "translation": "Wind God"},
                 {"source": FENGSHEN_ZHAN, "translation": "Divine Wind Slash"}],
                [(FENGSHEN, FENGSHEN_ZHAN)],
            )
        self.assertEqual(outcome.inconsistent, [])
        self.assertEqual(outcome.judged, 0)
        post.assert_not_called()

    def test_defaults_reproduce_the_behaviour_the_tests_already_pinned(self):
        outcome = run_sources(
            [term(FENGSHEN, "character"), term(FENGSHEN_ZHAN)],
            relations={(FENGSHEN, FENGSHEN_ZHAN): "subordinate"},
        )
        self.assertEqual(outcome.keep_separate, {(FENGSHEN, FENGSHEN_ZHAN)})


class TestPromptCatalogue(unittest.TestCase):
    CONFIG = {"target_language": "chinese_simplified"}

    def test_both_stages_offer_a_general_and_a_decision_variant(self):
        for stage, decision_id in ((BASIC, PromptBuilderEnum.EXTRACT_COMMON_DECISION),
                                   (JUDGMENT, PromptBuilderEnum.EXTRACT_JUDGMENT_DECISION)):
            with self.subTest(stage=stage):
                presets = PromptBuilderExtraction.get_system_presets(self.CONFIG, stage)
                self.assertEqual(len(presets), 2)
                self.assertEqual([p["key"] for p in presets],
                                 [PromptBuilderExtraction.COMMON_PRESET_KEY,
                                  PromptBuilderExtraction.DECISION_PRESET_KEY])
                self.assertEqual(presets[1]["id"], decision_id)
                self.assertTrue(presets[0]["content"].strip())
                self.assertTrue(presets[1]["content"].strip())
                self.assertNotEqual(presets[0]["content"], presets[1]["content"])
                self.assertTrue(all(p["type"] == "system" for p in presets))

    def test_the_decision_variant_follows_the_target_language(self):
        zh = PromptBuilderExtraction.get_system_presets(self.CONFIG, BASIC)[1]["content"]
        en = PromptBuilderExtraction.get_system_presets(
            {"target_language": "english"}, BASIC)[1]["content"]
        self.assertNotEqual(zh, en)
        # The Chinese prompts name it 判定层 throughout; the English say "decision layer".
        self.assertIn("\u5224\u5b9a\u5c42", zh)
        self.assertIn("decision", en.lower())

    def test_a_selected_built_in_variant_actually_takes_effect(self):
        """The card existed but build_system ignored it - a UI-only setting."""
        presets = PromptBuilderExtraction.get_system_presets(self.CONFIG, BASIC)
        config = dict(self.CONFIG)
        config[PromptBuilderExtraction.STAGES[BASIC]["selection_key"]] = {
            "last_selected_id": presets[1]["id"],
            "prompt_content": presets[1]["content"],
        }
        self.assertEqual(PromptBuilderExtraction.build_system(config, BASIC), presets[1]["content"])

    def test_the_general_variant_still_takes_effect_when_selected(self):
        presets = PromptBuilderExtraction.get_system_presets(self.CONFIG, BASIC)
        config = dict(self.CONFIG)
        config[PromptBuilderExtraction.STAGES[BASIC]["selection_key"]] = {
            "last_selected_id": presets[0]["id"],
            "prompt_content": presets[0]["content"],
        }
        self.assertEqual(PromptBuilderExtraction.build_system(config, BASIC), presets[0]["content"])

    def test_no_selection_falls_back_to_the_general_variant(self):
        self.assertEqual(
            PromptBuilderExtraction.build_system(dict(self.CONFIG), BASIC),
            PromptBuilderExtraction.get_system_default(self.CONFIG, BASIC),
        )

    def test_a_user_prompt_still_wins_over_every_built_in(self):
        config = dict(self.CONFIG)
        settings = PromptBuilderExtraction.STAGES[JUDGMENT]
        config[settings["user_data_key"]] = [
            {"id": "u1", "name": "mine", "content": "USER CONTENT", "type": "user"},
        ]
        config[settings["selection_key"]] = {"last_selected_id": "u1", "prompt_content": "USER CONTENT"}
        self.assertEqual(PromptBuilderExtraction.build_system(config, JUDGMENT), "USER CONTENT")

    def test_the_decision_files_are_actually_present_and_distinct(self):
        for stage in (BASIC, JUDGMENT):
            for language in ("zh", "en"):
                with self.subTest(stage=stage, language=language):
                    content = PromptBuilderExtraction._read_decision_file(stage, language)
                    self.assertGreater(len(content), 500, "a real system prompt, not a stub")
                    self.assertIn("json", content.lower())




class TestUpsert(unittest.TestCase):
    """Editing a settings card must not move it, and must not duplicate it."""

    def test_a_new_entry_is_appended(self):
        result = DecisionSettings.upsert_user_entry([], {"id": "a", "name": "A"})
        self.assertEqual([e["id"] for e in result], ["a"])

    def test_an_existing_entry_is_replaced_in_place(self):
        entries = [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}, {"id": "c", "name": "C"}]
        result = DecisionSettings.upsert_user_entry(entries, {"id": "b", "name": "B2"})
        self.assertEqual([e["id"] for e in result], ["a", "b", "c"], "order is preserved")
        self.assertEqual(result[1]["name"], "B2")
        self.assertEqual(len(result), 3, "no duplicate row")

    def test_the_input_list_is_not_mutated(self):
        entries = [{"id": "a", "name": "A"}]
        DecisionSettings.upsert_user_entry(entries, {"id": "b", "name": "B"})
        self.assertEqual([e["id"] for e in entries], ["a"])

    def test_junk_entries_are_carried_through_untouched(self):
        result = DecisionSettings.upsert_user_entry([None, "x", {"id": "a"}], {"id": "a", "name": "A"})
        self.assertEqual(result[0], None)
        self.assertEqual(result[1], "x")
        self.assertEqual(result[2]["name"], "A")

    def test_no_entries_at_all_is_fine(self):
        self.assertEqual([e["id"] for e in DecisionSettings.upsert_user_entry(None, {"id": "a"})], ["a"])


class TestSummaryTranslation(unittest.TestCase):
    def test_the_summary_goes_through_the_translator(self):
        text = DecisionSettings.summarize({}, translate=lambda _: "T:{0}-{1}")
        self.assertTrue(text.startswith("T:"), text)

    def test_without_a_translator_the_source_text_is_used(self):
        self.assertTrue(DecisionSettings.summarize({}).startswith("\u5173\u7cfb\u5224\u5b9a"))

    def test_the_summary_labels_are_translated_too(self):
        seen = []
        DecisionSettings.summarize({"relation_switch": False}, translate=lambda text: seen.append(text) or text)
        self.assertIn("\u5f00", seen)
        self.assertIn("\u5173", seen)



class TestDecisionPromptContent(unittest.TestCase):
    """The four decision prompts ARE the feature, so their distinctive rules are pinned.

    Copying a general prompt over a decision file would keep every JSON-contract test passing
    while silently removing the whole point. These markers are what makes the variant different.
    """

    ZH = {"target_language": "chinese_simplified"}
    EN = {"target_language": "english"}

    RULES = (
        (BASIC, "zh", ("\u63d0\u53ca\u539f\u5b50\u5316",          # atomic mentions
                       "\u4fdd\u7559\u72ec\u7acb\u77ed\u5f62\u5f0f",  # keep standalone short forms
                       "\u62d2\u7edd\u4e0d\u5b89\u5168\u952e",        # refuse unsafe keys
                       "\u4e0b\u6e38\u5224\u5b9a\u5c42")),             # the decision layer itself
        (BASIC, "en", ("Atomic mentions", "Keep standalone short forms",
                       "Refuse unsafe keys", "decision layer")),
        (JUDGMENT, "zh", ("\u533a\u5206\u540c\u4e00\u5b9e\u4f53\u4e0e\u6d3e\u751f\u5b9e\u4f53",
                          "\u5171\u4eab\u540d\u79f0\u90e8\u5206\u8bd1\u6cd5\u4e00\u81f4",
                          "\u4e3b\u952e\u9010\u5b57\u53d6\u7528",
                          "\u62d2\u7edd\u4e0d\u5b89\u5168\u952e")),
        (JUDGMENT, "en", ("Tell the same entity from a derived one",
                          "Render the shared name part consistently",
                          "Take the source verbatim", "Refuse unsafe keys")),
    )

    def general_text(self, stage, language):
        config = self.ZH if language == "zh" else self.EN
        return PromptBuilderExtraction.get_system_presets(config, stage)[0]["content"]

    def test_each_decision_prompt_carries_its_distinctive_rules(self):
        for stage, language, markers in self.RULES:
            text = PromptBuilderExtraction._read_decision_file(stage, language)
            for marker in markers:
                with self.subTest(stage=stage, language=language, marker=marker):
                    self.assertIn(marker, text)

    def test_the_general_prompt_does_not_carry_the_decision_rules(self):
        """If it did, the two variants would have converged and one of them is redundant."""
        for stage, language, markers in self.RULES:
            text = self.general_text(stage, language)
            for marker in markers:
                with self.subTest(stage=stage, language=language, marker=marker):
                    self.assertNotIn(marker, text)

    # Stage 1 also emits non_translate; stage 2 only decides characters and terms.
    CONTRACT_FIELDS = {
        BASIC: ("source", "recommended_translation", "gender", "category_path", "note", "marker"),
        JUDGMENT: ("source", "recommended_translation", "gender", "category_path", "note"),
    }

    def test_the_decision_prompt_keeps_the_json_contract_the_code_parses(self):
        for stage, language, _ in self.RULES:
            with self.subTest(stage=stage, language=language):
                text = PromptBuilderExtraction._read_decision_file(stage, language)
                for field in self.CONTRACT_FIELDS[stage]:
                    self.assertIn(field, text)


class TestUiStringsAreLocalized(unittest.TestCase):
    """A label with no entry renders as Chinese inside an English UI.

    The strings are taken from the code constants rather than copied, so adding a new setting
    or preset name without translating it fails here instead of shipping half-translated.
    """

    # Labels written directly in the settings page.
    PAGE_STRINGS = (
        "\u51b3\u7b56\u8bbe\u7f6e",              # Decision Settings (nav + section title)
        "\u9ed8\u8ba4\uff08\u63a8\u8350\uff09",  # Default (recommended)
        "\u5f53\u524d\u51b3\u7b56\u8bbe\u7f6e",  # Current decision settings
        "\u521b\u5efa\u65b0\u8bbe\u7f6e",        # Create new settings
        "\u540d\u79f0",                            # Name
        "\u8bf7\u8f93\u5165\u540d\u79f0",        # Enter a name
        "\u4f8b\u5982\uff1a\u53ea\u62a5\u544a\u4e0d\u5254\u9664",  # e.g. report only
    )

    @classmethod
    def setUpClass(cls):
        folder = Path(__file__).parents[1] / "Resource" / "Localization"
        strings = {}
        for path in sorted(folder.glob("*.json")):
            for part in json.loads(path.read_text(encoding="utf-8")).values():
                if isinstance(part, dict):
                    strings.update(part)
        cls.strings = strings

    def assert_localized(self, label: str, context: str) -> None:
        self.assertIn(label, self.strings, "no translation entry for {0}: {1!r}".format(context, label))
        self.assertEqual(
            sorted(self.strings[label]),
            ["English", "\u65e5\u672c\u8a9e", "\u7b80\u4e2d", "\u7e41\u4e2d"],
            "{0} is missing a language".format(context),
        )

    def test_every_decision_field_label_is_localized(self):
        for key, label in DecisionSettings.FIELD_LABELS.items():
            with self.subTest(key=key):
                self.assert_localized(label, "field " + key)

    def test_every_preset_name_and_description_is_localized(self):
        pairs = list(PromptBuilderExtraction.PRESET_NAMES.items())
        pairs += list(PromptBuilderExtraction.PRESET_DESCRIPTIONS.items())
        for key, value in pairs:
            with self.subTest(key=key):
                self.assert_localized(value, "preset " + str(key))

    def test_the_on_off_labels_the_summary_prints_are_localized(self):
        self.assert_localized("\u5f00", "summary on")
        self.assert_localized("\u5173", "summary off")

    def test_the_summary_format_survives_every_translation(self):
        """A translation that drops {5:.2f} would raise IndexError inside the card."""
        self.assert_localized(DecisionSettings.SUMMARY_FORMAT, "summary format")
        for language, text in self.strings[DecisionSettings.SUMMARY_FORMAT].items():
            with self.subTest(language=language):
                rendered = DecisionSettings.summarize({}, translate=lambda _: text)
                self.assertIn("0.50", rendered)

    def test_the_settings_page_labels_are_localized(self):
        for label in self.PAGE_STRINGS:
            with self.subTest(label=label):
                self.assert_localized(label, "page")


if __name__ == "__main__":
    unittest.main()