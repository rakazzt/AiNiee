"""The language verdict: what the decision model is asked, and how its answer is read.

This is the case a character rule cannot decide. For a Chinese target, Han characters are
legitimate in a correct translation, so comparing character classes can never separate
"translated" from "not translated" - which is why that check is disabled for zh->ja and
zh->tw today. Two noul questions can separate them, and both failures matter differently:
a line can be in the wrong language without being untranslated, and it can be untranslated
while looking exactly like the target language.

Everything here is Qt-free, so it runs everywhere; the wiring into LanguageChecker is
covered by test_language_decision.py, which needs the heavy reader dependencies.
"""
import unittest

from ModuleFolders.Infrastructure.DecisionEngine import Checks


def answers(in_target=None, still_source=None, index=0):
    lang_id, residual_id = Checks.language_question_ids(index)
    result = {}
    if in_target is not None:
        result[lang_id] = {"type": "noul", "noul": in_target}
    if still_source is not None:
        result[residual_id] = {"type": "noul", "noul": still_source}
    return result


class TestQuestions(unittest.TestCase):
    def test_two_atomic_questions_not_one_compound_one(self):
        questions = Checks.language_questions(0)
        self.assertEqual(sorted(questions), ["i0_lang", "i0_residual"])
        for question in questions.values():
            self.assertEqual(question["type"], "noul")

    def test_questions_point_at_their_own_item_by_path(self):
        questions = Checks.language_questions(7)
        self.assertIn("items.7.translated", questions["i7_lang"]["instructions"])
        self.assertIn("items.7.translated", questions["i7_residual"]["instructions"])
        self.assertIn("items.7.source", questions["i7_residual"]["instructions"])
        # The language itself is a state value, not baked into the sentence.
        self.assertIn("target_language", questions["i7_lang"]["instructions"])

    def test_a_question_is_asked_per_index(self):
        self.assertEqual(sorted(Checks.language_questions(3)), ["i3_lang", "i3_residual"])


class TestVerdict(unittest.TestCase):
    def verdict(self, index=0, **probabilities):
        return Checks.language_verdict(answers(index=index, **probabilities), index, "ja", "zh")

    def test_text_in_the_target_language_passes(self):
        self.assertEqual(self.verdict(in_target=0.97, still_source=0.02), (["ja"], 0.97))

    def test_untranslated_source_text_is_named_as_the_source_language(self):
        """The judgment the character rule has to skip for zh -> ja."""
        self.assertEqual(self.verdict(in_target=0.04, still_source=0.96), (["zh"], 0.96))

    def test_a_third_language_is_unknown_and_keeps_the_confidence_of_that_reading(self):
        language, confidence = self.verdict(in_target=0.08, still_source=0.10)
        self.assertEqual(language, ["unknown"])
        self.assertAlmostEqual(confidence, 0.92)

    def test_the_threshold_is_inclusive(self):
        self.assertEqual(self.verdict(in_target=0.5)[0], ["ja"])
        self.assertEqual(self.verdict(in_target=0.499, still_source=0.9)[0], ["zh"])

    def test_a_high_residual_probability_is_ignored_when_the_line_is_in_target(self):
        # Quoted source fragments are legitimate; the language question wins.
        self.assertEqual(self.verdict(in_target=0.9, still_source=0.8)[0], ["ja"])

    def test_a_missing_language_answer_is_unknown_not_a_failure(self):
        """Fail open: reading this as "no" would flag every line on an outage."""
        self.assertIsNone(Checks.language_verdict(answers(still_source=0.99), 0, "ja", "zh"))
        self.assertIsNone(Checks.language_verdict({}, 0, "ja", "zh"))
        self.assertIsNone(Checks.language_verdict(None, 0, "ja", "zh"))

    def test_a_missing_residual_answer_still_fails_the_line_but_not_the_model(self):
        language, confidence = self.verdict(in_target=0.02)
        self.assertEqual(language, ["unknown"])
        self.assertAlmostEqual(confidence, 0.98)

    def test_a_blank_source_language_falls_back_to_unknown_rather_than_empty(self):
        result = Checks.language_verdict(answers(in_target=0.01, still_source=0.99), 0, "ja", "")
        self.assertEqual(result[0], ["unknown"])

    def test_wrong_typed_answers_are_treated_as_missing(self):
        broken = {"i0_lang": {"type": "choice", "choice": "ja", "probabilities": {"ja": 1.0}},
                  "i0_residual": {"type": "noul", "noul": "yes"}}
        self.assertIsNone(Checks.language_verdict(broken, 0, "ja", "zh"))

    def test_verdicts_are_independent_across_a_batch(self):
        merged = {}
        for index, probabilities in enumerate([(0.9, 0.05), (0.02, 0.95), (0.05, 0.1)]):
            merged.update(answers(in_target=probabilities[0], still_source=probabilities[1], index=index))
        verdicts = [Checks.language_verdict(merged, i, "ja", "zh") for i in range(3)]
        self.assertEqual([v[0] for v in verdicts], [["ja"], ["zh"], ["unknown"]])


if __name__ == "__main__":
    unittest.main()