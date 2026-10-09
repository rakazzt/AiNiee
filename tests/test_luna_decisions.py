"""GPT-6 Luna Decisions: the typed helper, tested without ever touching the network.

Five properties carry this module, so each is pinned here:

1. **The body is the OpenRouter one.** That route requires model + state + questions and
   rejects the OpenAI input part shapes, so an "input" field or a message list on the wire
   is a silent 400 in production. The tests assert the exact keys sent.
2. **Answers are read by position.** The route returns a map keyed by question name, so a
   reader that loses the order would report the wrong decision for the wrong question.
3. **An unreadable answer is unknown, never "no".** A missing or malformed answer must not
   read as 0.0, or an outage looks like a confident negative.
4. **A refusal is a refusal.** It is not a value and not an absence.
5. **One attempt, bounded.** No retry and no Idempotency-Key: a timed-out decision may
   already be billed. Every test asserts urlopen was called exactly once.

The transport is mocked at urlopen, so no key and no credits are needed to run this.
"""
import json
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError

from ModuleFolders.Infrastructure.DecisionEngine import DecisionEngine, LunaDecisions, Questions, SystemOneClient
from ModuleFolders.Infrastructure.DecisionEngine.SystemOneClient import DecisionError

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "openai/gpt-6-luna-decisions"
DATA_URL = "data:image/png;base64,iVBORw0KGgo="


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


def body(**answers):
    return {
        "model": MODEL,
        "answers": answers,
        "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.000019992},
        "id": "gen-dec-1",
        "provider": "OpenAI",
    }


def client(**overrides):
    options = dict(api_key="secret", timeout=45)
    options.update(overrides)
    return LunaDecisions.LunaDecisions(**options)


def patch_urlopen(return_value=None, side_effect=None):
    return mock.patch.object(
        SystemOneClient.urlrequest, "urlopen",
        return_value=return_value, side_effect=side_effect,
    )


def sent(post):
    """The request object of the single POST that was made."""
    return post.call_args[0][0]


class TestQuestionBuilders(unittest.TestCase):
    def test_predicate_is_a_noul_without_criteria(self):
        self.assertEqual(LunaDecisions.predicate("Is it risky?"),
                         {"type": "predicate", "instructions": "Is it risky?"})

    def test_a_name_is_carried_only_when_given(self):
        self.assertEqual(LunaDecisions.predicate("ok?", name="review")["name"], "review")
        self.assertNotIn("name", LunaDecisions.predicate("ok?"))

    def test_choice_keeps_values_and_descriptions(self):
        question = LunaDecisions.choice("Which?", [
            {"value": "accept", "description": "Safe to merge."},
            {"value": "reject", "description": "Needs work."},
        ])
        self.assertEqual(question["choices"], [
            {"value": "accept", "description": "Safe to merge."},
            {"value": "reject", "description": "Needs work."},
        ])

    def test_choice_rejects_a_typo_rather_than_dropping_it(self):
        with self.assertRaises(Questions.QuestionError):
            LunaDecisions.choice("Which?", [{"value": "a", "descrition": "typo"}])

    def test_choice_rejects_duplicates_and_non_lists(self):
        for bad in ([{"value": "a"}, {"value": "a"}], {}, [], "a"):
            with self.assertRaises(Questions.QuestionError, msg=repr(bad)):
                LunaDecisions.choice("Which?", bad)

    def test_score_keeps_level_order_and_folds_in_the_label(self):
        question = LunaDecisions.score("How bad?", [
            {"label": "Cosmetic", "description": "Appearance only."},
            {"label": "Blocked", "description": "No workaround."},
        ])
        self.assertEqual([level["label"] for level in question["levels"]], ["Cosmetic", "Blocked"])

    def test_score_rejects_a_level_count_the_api_would_reject(self):
        with self.assertRaises(Questions.QuestionError):
            LunaDecisions.score("How bad?", [{"label": "Only one"}])


class TestWireShape(unittest.TestCase):
    def test_the_body_is_the_openrouter_one(self):
        """model + state + questions, and never the OpenAI "input" field."""
        questions = [LunaDecisions.predicate("Is it risky?", name="risk")]
        with patch_urlopen(return_value=FakeResponse(body(risk={"type": "noul", "noul": 0.9}))) as post:
            client().ask("the evidence", questions)
        payload = json.loads(sent(post).data.decode("utf-8"))
        self.assertEqual(sorted(payload), ["model", "questions", "state"])
        self.assertEqual(payload["model"], MODEL)
        self.assertEqual(payload["state"], "the evidence")
        self.assertEqual(payload["questions"],
                         {"risk": {"type": "noul", "instructions": "Is it risky?"}})

    def test_the_endpoint_is_the_decisions_router(self):
        with patch_urlopen(return_value=FakeResponse(body(risk={"type": "noul", "noul": 0.9}))) as post:
            client().ask("x", [LunaDecisions.predicate("ok?", name="risk")])
        self.assertEqual(sent(post).full_url, ENDPOINT)

    def test_an_unnamed_question_gets_a_positional_key(self):
        with patch_urlopen(return_value=FakeResponse(body(q0={"type": "noul", "noul": 0.5}))) as post:
            decision = client().ask("x", [LunaDecisions.predicate("ok?")])
        payload = json.loads(sent(post).data.decode("utf-8"))
        self.assertEqual(list(payload["questions"]), ["q0"])
        self.assertIsNone(decision.answers[0].name)

    def test_duplicate_names_are_refused(self):
        with self.assertRaises(Questions.QuestionError):
            LunaDecisions._encode([
                LunaDecisions.predicate("a", name="same"),
                LunaDecisions.predicate("b", name="same"),
            ])

    def test_a_name_the_route_would_reject_is_refused(self):
        with self.assertRaises(Questions.QuestionError):
            LunaDecisions._encode([LunaDecisions.predicate("a", name="has space")])

    def test_unknown_type_and_typo_field_are_refused(self):
        with patch_urlopen() as post:
            with self.assertRaises(Questions.QuestionError):
                client().ask("x", [{"type": "yesno", "instructions": "ok?"}])
            with self.assertRaises(Questions.QuestionError):
                client().ask("x", [{"type": "predicate", "instructions": "ok?", "critera": {}}])
        post.assert_not_called()


class TestState(unittest.TestCase):
    def test_a_message_list_becomes_plain_text_and_image_parts(self):
        state = LunaDecisions.encode_state([{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Inspect this diff."},
                {"type": "input_image", "image_url": DATA_URL},
            ],
        }])
        self.assertEqual(state, [
            "Inspect this diff.",
            {"type": "image_url", "image_url": {"url": DATA_URL}},
        ])

    def test_a_remote_image_url_is_refused_because_it_is_never_fetched(self):
        with self.assertRaises(Questions.QuestionError):
            LunaDecisions.encode_state([{"role": "user", "content": [
                {"type": "input_image", "image_url": "https://example.com/a.png"},
            ]}])

    def test_a_file_id_is_refused(self):
        with self.assertRaises(Questions.QuestionError):
            LunaDecisions.encode_state([{"type": "input_image", "file_id": "file-1"}])

    def test_an_unsupported_part_type_is_refused(self):
        with self.assertRaises(Questions.QuestionError):
            LunaDecisions.encode_state([{"type": "input_file", "file_id": "f"}])

    def test_more_images_than_the_documented_cap_are_refused(self):
        parts = [{"type": "input_image", "image_url": DATA_URL}] * (LunaDecisions.MAX_IMAGES + 1)
        with self.assertRaises(Questions.QuestionError):
            LunaDecisions.encode_state(parts)

    def test_a_plain_string_and_an_object_pass_through(self):
        self.assertEqual(LunaDecisions.encode_state("plain"), "plain")
        self.assertEqual(LunaDecisions.encode_state({"a": 1}), {"a": 1})

    def test_empty_evidence_is_refused(self):
        with self.assertRaises(Questions.QuestionError):
            LunaDecisions.encode_state([])

    def test_review_state_carries_the_diff_and_the_test_summary(self):
        text = LunaDecisions.review_state("- a\n+ b", "12 tests pass")
        self.assertIn("- a", text)
        self.assertIn("12 tests pass", text)

    def test_review_state_is_truncated_rather_than_sent_whole(self):
        whole = LunaDecisions.review_state("x" * 500, "ok")
        text = LunaDecisions.review_state("x" * 500, "ok", max_chars=100)
        self.assertTrue(whole.startswith(text[:100]))
        self.assertLess(len(text), len(whole))
        self.assertIn("truncated", text)


class TestReadingAnswers(unittest.TestCase):
    def test_answers_are_read_in_question_order(self):
        questions = [
            LunaDecisions.predicate("Risky?", name="risk"),
            LunaDecisions.choice("Verdict?", [{"value": "accept"}, {"value": "reject"}], name="verdict"),
            LunaDecisions.score("Severity?", [{"label": "Low"}, {"label": "High"}], name="severity"),
        ]
        payload = body(
            severity={"type": "score", "score": 1.99, "confidence": 0.99,
                      "probabilities": {"0": 0.0, "1": 0.01, "2": 0.99}, "legend": {"0": "Low", "1": "High"}},
            verdict={"type": "choice", "choice": "reject", "confidence": 0.95,
                     "probabilities": {"accept": 0.05, "reject": 0.95}},
            risk={"type": "noul", "noul": 0.87},
        )
        with patch_urlopen(return_value=FakeResponse(payload)):
            decision = client().ask("evidence", questions)
        self.assertEqual([answer.type for answer in decision.answers],
                         ["predicate", "choice", "score"])
        self.assertAlmostEqual(decision.answer_at(0).value, 0.87)
        self.assertEqual(decision.answer_at(1).value, "reject")
        self.assertEqual(decision.answer_at(1).name, "verdict")
        self.assertAlmostEqual(decision.answer_at(2).value, 1.99)
        self.assertEqual(decision.answer_at(2).legend, {"0": "Low", "1": "High"})

    def test_the_review_question_is_read_at_its_own_position(self):
        "The failure this guards: reading position 0 when the question was second."
        questions = [
            LunaDecisions.predicate("Is it formatted?", name="format"),
            LunaDecisions.predicate("Does it introduce a bug?", name="review"),
        ]
        with patch_urlopen(return_value=FakeResponse(body(
            format={"type": "noul", "noul": 0.99},
            review={"type": "noul", "noul": 0.04},
        ))):
            decision = client().ask("diff", questions)
        self.assertAlmostEqual(decision.answer_at(1).value, 0.04)
        self.assertEqual(decision.answer_at(1).name, "review")

    def test_a_refusal_is_reported_as_a_refusal(self):
        with patch_urlopen(return_value=FakeResponse(body(
                review={"type": "refusal", "name": "review"}))):
            decision = client().ask("diff", [LunaDecisions.predicate("Bug?", name="review")])
        self.assertTrue(decision.answers[0].is_refusal)
        self.assertIsNone(decision.answers[0].value)
        self.assertFalse(decision.answers[0].is_available)

    def test_a_missing_answer_is_unavailable_and_never_zero(self):
        with patch_urlopen(return_value=FakeResponse(body())):
            decision = client().ask("diff", [LunaDecisions.predicate("Bug?", name="review")])
        self.assertEqual(decision.answers[0].type, LunaDecisions.UNAVAILABLE)
        self.assertIsNone(decision.answers[0].value)
        self.assertNotEqual(decision.answers[0].value, 0.0)

    def test_a_malformed_answer_is_unavailable(self):
        with patch_urlopen(return_value=FakeResponse(body(review={"type": "noul", "noul": "yes"}))):
            decision = client().ask("diff", [LunaDecisions.predicate("Bug?", name="review")])
        self.assertEqual(decision.answers[0].type, LunaDecisions.UNAVAILABLE)

    def test_a_choice_outside_the_offered_options_is_unavailable(self):
        with patch_urlopen(return_value=FakeResponse(body(
                verdict={"type": "choice", "choice": "maybe", "probabilities": {"accept": 1.0}}))):
            decision = client().ask("diff", [
                LunaDecisions.choice("Verdict?", [{"value": "accept"}], name="verdict")])
        self.assertEqual(decision.answers[0].type, LunaDecisions.UNAVAILABLE)

    def test_an_out_of_range_position_has_no_answer(self):
        with patch_urlopen(return_value=FakeResponse(body(review={"type": "noul", "noul": 0.5}))):
            decision = client().ask("diff", [LunaDecisions.predicate("Bug?", name="review")])
        self.assertIsNone(decision.answer_at(1))
        self.assertIsNone(decision.answer_at(-1))

    def test_the_reported_usage_is_kept(self):
        with patch_urlopen(return_value=FakeResponse(body(review={"type": "noul", "noul": 0.5}))):
            decision = client().ask("diff", [LunaDecisions.predicate("Bug?", name="review")])
        self.assertEqual(decision.usage["input_tokens"], 476)
        self.assertEqual(decision.usage["output_tokens"], 70)
        self.assertAlmostEqual(decision.usage["cost"], 0.000019992)
        self.assertEqual(decision.model, MODEL)
        self.assertIn("476 input tokens", decision.usage_line())

    def test_describe_prints_one_line_per_answer(self):
        with patch_urlopen(return_value=FakeResponse(body(
                review={"type": "noul", "noul": 0.04},
                verdict={"type": "choice", "choice": "reject", "confidence": 0.95}))):
            decision = client().ask("diff", [
                LunaDecisions.predicate("Bug?", name="review"),
                LunaDecisions.choice("Verdict?", [{"value": "reject"}], name="verdict"),
            ])
        lines = decision.describe()
        self.assertEqual(len(lines), 2)
        self.assertIn("probability 0.0400", lines[0])
        self.assertIn("reject", lines[1])


class TestTransportPolicy(unittest.TestCase):
    def test_one_attempt_only(self):
        with patch_urlopen(side_effect=URLError("down")) as post:
            with self.assertRaises(DecisionError):
                client().ask("diff", [LunaDecisions.predicate("Bug?", name="review")])
        self.assertEqual(post.call_count, 1)

    def test_no_idempotency_key_is_sent(self):
        with patch_urlopen(return_value=FakeResponse(body(review={"type": "noul", "noul": 0.5}))) as post:
            client().ask("diff", [LunaDecisions.predicate("Bug?", name="review")])
        headers = {key.lower() for key in sent(post).headers}
        self.assertNotIn("idempotency-key", headers)
        self.assertIn("authorization", headers)

    def test_the_timeout_is_bounded_and_forwarded(self):
        with patch_urlopen(return_value=FakeResponse(body(review={"type": "noul", "noul": 0.5}))) as post:
            client(timeout=12).ask("diff", [LunaDecisions.predicate("Bug?", name="review")],
                                   timeout=7)
        self.assertEqual(post.call_args[1]["timeout"], 7)

    def test_the_configured_timeout_is_used_when_none_is_passed(self):
        with patch_urlopen(return_value=FakeResponse(body(review={"type": "noul", "noul": 0.5}))) as post:
            client(timeout=12).ask("diff", [LunaDecisions.predicate("Bug?", name="review")])
        self.assertEqual(post.call_args[1]["timeout"], 12)

    def test_an_http_error_is_surfaced_with_its_code_and_body(self):
        error = HTTPError(ENDPOINT, 402, "Payment Required", {},
                          FakeResponse(raw=b'{"error":{"message":"Insufficient credits"}}'))
        with patch_urlopen(side_effect=error):
            with self.assertRaises(DecisionError) as caught:
                client().ask("diff", [LunaDecisions.predicate("Bug?", name="review")])
        self.assertIn("402", str(caught.exception))
        self.assertIn("Insufficient credits", str(caught.exception))

    def test_a_transport_error_is_surfaced(self):
        with patch_urlopen(side_effect=URLError("connection refused")):
            with self.assertRaises(DecisionError):
                client().ask("diff", [LunaDecisions.predicate("Bug?", name="review")])

    def test_an_unreadable_body_is_surfaced(self):
        with patch_urlopen(return_value=FakeResponse(raw=b"not json")):
            with self.assertRaises(DecisionError):
                client().ask("diff", [LunaDecisions.predicate("Bug?", name="review")])

    def test_a_missing_answers_map_is_surfaced(self):
        with patch_urlopen(return_value=FakeResponse({"model": MODEL})):
            with self.assertRaises(DecisionError):
                client().ask("diff", [LunaDecisions.predicate("Bug?", name="review")])

    def test_a_missing_key_is_refused_before_any_request(self):
        with patch_urlopen() as post:
            with self.assertRaises(DecisionError):
                client(api_key="").ask("diff", [LunaDecisions.predicate("Bug?", name="review")])
        post.assert_not_called()


class TestShippedPreset(unittest.TestCase):
    """The wiring, end to end: the shipped entry must resolve to the documented route.

    A preset that declares a format the engine cannot resolve would silently fall back to
    the TypeSafe route and 404 against OpenRouter, so the URL is asserted rather than the
    declaration.
    """

    def setUp(self):
        preset_path = Path(__file__).parents[1] / "Resource" / "platforms" / "preset.json"
        self.platform = json.loads(preset_path.read_text(encoding="utf-8"))["platforms"]["luna_decisions"]

    def test_the_entry_is_a_decision_platform_for_this_model(self):
        self.assertEqual(self.platform["group"], "decision")
        self.assertEqual(self.platform["model"], MODEL)
        self.assertEqual(self.platform["api_format"], "decisions")

    def test_the_entry_resolves_to_the_decisions_router(self):
        engine = DecisionEngine.DecisionEngine.from_platform(dict(self.platform, api_key="secret"))
        self.assertEqual(engine.client.shape, SystemOneClient.DECISIONS)
        self.assertEqual(engine.client.endpoint, ENDPOINT)
        self.assertEqual(engine.client.model, MODEL)

    def test_the_entry_ships_without_a_key(self):
        self.assertEqual(self.platform["api_key"], "")


if __name__ == "__main__":
    unittest.main()
