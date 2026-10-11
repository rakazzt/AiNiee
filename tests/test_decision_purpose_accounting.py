"""Where the decision layer's money and its mistakes go, and what counts as generic.

Two things this pins. First, the sweep runs three different judgements - is a term too
generic, is a pair a derived name, is a shared name translated consistently - through one
engine, and a single blended total cannot say which of them is spending the money or
failing. Second, the generic judgement was dropping invented names that are plainly not
generic, so the question now says a proper noun is not one.
"""
import unittest

from ModuleFolders.Infrastructure.DecisionEngine import ConsistencySweep
from ModuleFolders.Infrastructure.DecisionEngine.DecisionEngine import DecisionEngine
from ModuleFolders.Infrastructure.DecisionEngine.SystemOneClient import DecisionError as RealDecisionError


class FakeClient:
    """Stands in for SystemOneClient: records what was asked, returns canned answers."""

    model = "openai/gpt-6-luna-decisions"
    endpoint = "https://openrouter.ai/api/alpha/decisions"
    shape = "decisions"

    def __init__(self, body=None, error=None):
        self.body = body if body is not None else {"answers": {"a": 1}}
        self.error = error
        self.calls = 0

    def evaluate(self, state, questions):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.body


# The engine catches SystemOneClient's own error type, so the double must raise that.
def offline_error():
    return RealDecisionError("model_unavailable", kind="http", status=503, detail="offline")


class TestPurposeAccounting(unittest.TestCase):
    def test_each_purpose_gets_its_own_counters(self):
        engine = DecisionEngine(FakeClient(body={"answers": {}, "usage": {
            "input_tokens": 100, "output_tokens": 0, "cost": 0.0001}}))
        engine.ask({}, {"g0": 1}, purpose="term_generic")
        engine.ask({}, {"r0": 1, "r1": 1}, purpose="term_relation")
        engine.ask({}, {"c0": 1}, purpose="term_generic")

        summary = engine.summary()["purposes"]
        self.assertEqual(summary["term_generic"]["calls"], 2)
        self.assertEqual(summary["term_generic"]["questions"], 2)
        self.assertEqual(summary["term_relation"]["calls"], 1)
        self.assertEqual(summary["term_relation"]["questions"], 2)

    def test_usage_lands_on_the_purpose_that_spent_it(self):
        engine = DecisionEngine(FakeClient(body={"answers": {}, "usage": {
            "input_tokens": 300, "output_tokens": 0, "cost": 0.03}}))
        engine.ask({}, {"a": 1}, purpose="term_generic")
        summary = engine.summary()["purposes"]
        self.assertEqual(summary["term_generic"]["input_tokens"], 300)
        self.assertEqual(engine.input_tokens, 300)

    def test_a_failure_is_counted_against_its_own_purpose(self):
        engine = DecisionEngine(FakeClient(error=offline_error()))
        engine.ask({}, {"a": 1}, purpose="term_generic")
        engine.ask({}, {"b": 1}, purpose="term_consistency")
        summary = engine.summary()["purposes"]
        self.assertEqual(summary["term_generic"]["failures"], 1)
        self.assertEqual(summary["term_consistency"]["failures"], 1)
        self.assertEqual(engine.failures, 2)

    def test_an_unnamed_purpose_is_still_accounted_for(self):
        engine = DecisionEngine(FakeClient())
        engine.ask({}, {"a": 1})
        self.assertIn("未标注用途", engine.summary()["purposes"])

    def test_a_single_purpose_needs_no_breakdown(self):
        engine = DecisionEngine(FakeClient())
        engine.ask({}, {"a": 1}, purpose="term_generic")
        self.assertEqual(engine.purpose_lines(), [])

    def test_the_breakdown_leads_with_the_most_expensive_purpose(self):
        cheap = DecisionEngine(FakeClient(body={"answers": {}, "usage": {
            "input_tokens": 10, "output_tokens": 0, "cost": 0.00001}}))
        cheap.ask({}, {"a": 1}, purpose="term_consistency")
        costly = DecisionEngine(FakeClient(body={"answers": {}, "usage": {
            "input_tokens": 900, "output_tokens": 0, "cost": 0.009}}))
        costly.ask({}, {"a": 1}, purpose="term_generic")
        # One engine, two purposes: emulate by sharing counters through a second ask.
        engine = DecisionEngine(FakeClient(body={"answers": {}, "usage": {
            "input_tokens": 900, "output_tokens": 0, "cost": 0.009}}))
        engine.ask({}, {"a": 1}, purpose="term_generic")
        engine.client = FakeClient(body={"answers": {}, "usage": {
            "input_tokens": 10, "output_tokens": 0, "cost": 0.00001}})
        engine.ask({}, {"b": 1}, purpose="term_consistency")
        lines = engine.purpose_lines()
        self.assertEqual(len(lines), 2)
        self.assertIn("term_generic", lines[0])
        self.assertIn("term_consistency", lines[1])


class TestGenericQuestion(unittest.TestCase):
    def question_text(self):
        return ConsistencySweep.generic_question(0)["g0"]["instructions"]

    def test_a_proper_noun_is_declared_not_generic(self):
        text = self.question_text()
        self.assertIn("proper noun", text.lower())
        self.assertIn("NOT generic", text)

    def test_the_original_guard_is_still_asked_for(self):
        text = self.question_text()
        # The protections that were already working must survive the sharpening.
        self.assertIn("single character", text)
        self.assertIn("honorific", text)


if __name__ == "__main__":
    unittest.main()
