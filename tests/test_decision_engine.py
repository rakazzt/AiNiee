"""Decision layer (TypeSafe System One / JEV) regression tests.

Three properties carry the whole design, so each is pinned here:

1. **The wire format is the documented one.** Four provider surfaces, one request
   shape. A URL or body that drifts is a silent 400 in production.
2. **An unreadable answer is unknown, never "no".** The readers return None, and the
   engine fails open, because a classifier outage that read as "everything is wrong"
   would trigger a retry storm and cost real money.
3. **Batching does not change the answer paths.** Questions name values as
   items.<index>.field, so chunking must keep the original index, not a local one.
"""
import json
import unittest
from unittest import mock
from urllib.error import HTTPError, URLError

from ModuleFolders.Infrastructure.DecisionEngine import DecisionEngine, Questions, SystemOneClient
from ModuleFolders.Infrastructure.DecisionEngine.SystemOneClient import DecisionError


class FakeResponse:
    """Minimal stand-in for the object urlopen yields (a context manager)."""

    def __init__(self, payload=None, raw=None):
        self._raw = raw if raw is not None else json.dumps(payload).encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def answers_body(**answers):
    return {"model": "jev-1.13", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 2}}


def sent_body(request):
    return json.loads(request.data.decode("utf-8"))


def make_client(**overrides):
    options = dict(api_key="secret", model="jev-1.13", base_url="https://api.typesafe.ai")
    options.update(overrides)
    return SystemOneClient.SystemOneClient(**options)


PATCH = "urlopen"


class TestQuestionBuilders(unittest.TestCase):
    def test_noul_without_criteria(self):
        self.assertEqual(Questions.noul("Is it urgent?"),
                         {"type": "noul", "instructions": "Is it urgent?"})

    def test_noul_criteria_use_documented_true_false_keys(self):
        question = Questions.noul("Is it urgent?", true_meaning="Time sensitive", false_meaning="No")
        self.assertEqual(question["criteria"], {"true": "Time sensitive", "false": "No"})

    def test_choice_keeps_every_option_and_allows_null_description(self):
        question = Questions.choice("Which team?", {"billing": "Payments", "other": None})
        self.assertEqual(question["type"], "choice")
        self.assertEqual(question["criteria"], {"billing": "Payments", "other": None})

    def test_score_keeps_level_order(self):
        question = Questions.score("How bad?", ["Calm", "Frustrated", "Angry"])
        self.assertEqual(question["criteria"], ["Calm", "Frustrated", "Angry"])

    def test_instructions_may_be_structured(self):
        question = Questions.noul({"question": "Same person?", "record": {"name": "A"}})
        self.assertIsInstance(question["instructions"], dict)


class TestValidation(unittest.TestCase):
    def test_rejects_unknown_field_so_a_typo_cannot_change_the_question(self):
        with self.assertRaises(Questions.QuestionError):
            Questions.validate({"q": {"type": "noul", "instructions": "ok", "critera": {}}})

    def test_rejects_unknown_type(self):
        with self.assertRaises(Questions.QuestionError):
            Questions.validate({"q": {"type": "yesno", "instructions": "ok"}})

    def test_rejects_bad_question_id(self):
        for bad in ("has space", "", "x" * 101, "emoji\u2728"):
            with self.assertRaises(Questions.QuestionError, msg=bad):
                Questions.validate({bad: Questions.noul("ok?")})

    def test_accepts_documented_id_characters(self):
        Questions.validate({"item_0.lang-ok": Questions.noul("ok?")})

    def test_rejects_blank_instructions(self):
        with self.assertRaises(Questions.QuestionError):
            Questions.validate({"q": Questions.noul("   ")})

    def test_rejects_more_questions_than_the_provider_accepts(self):
        too_many = {"q%d" % i: Questions.noul("ok?") for i in range(Questions.MAX_QUESTIONS + 1)}
        with self.assertRaises(Questions.QuestionError):
            Questions.validate(too_many)

    def test_rejects_empty_question_map(self):
        with self.assertRaises(Questions.QuestionError):
            Questions.validate({})

    def test_score_level_bounds_are_enforced(self):
        with self.assertRaises(Questions.QuestionError):
            Questions.score("rate", ["only one"])
        with self.assertRaises(Questions.QuestionError):
            Questions.score("rate", ["l%d" % i for i in range(Questions.MAX_SCORE_LEVELS + 1)])
        Questions.score("rate", ["a", "b"])  # the documented minimum

    def test_choice_option_bounds_are_enforced(self):
        options = {"o%d" % i: None for i in range(Questions.MAX_CHOICE_OPTIONS + 1)}
        with self.assertRaises(Questions.QuestionError):
            Questions.choice("pick", options)

    def test_choice_and_score_require_criteria(self):
        with self.assertRaises(Questions.QuestionError):
            Questions.validate({"q": {"type": "choice", "instructions": "pick"}})
        with self.assertRaises(Questions.QuestionError):
            Questions.validate({"q": {"type": "score", "instructions": "rate"}})


class TestAnswerReaders(unittest.TestCase):
    def test_missing_answer_reads_as_unknown_not_zero(self):
        # Load-bearing: 0.0 would mean "definitely no".
        self.assertIsNone(Questions.noul_probability({}, "q"))
        self.assertIsNone(Questions.noul_probability(None, "q"))
        self.assertIsNone(Questions.noul_probability({"q": {"type": "choice"}}, "q"))
        self.assertIsNone(Questions.noul_probability({"q": {"type": "noul"}}, "q"))

    def test_noul_probability_is_read_and_clamped(self):
        self.assertEqual(Questions.noul_probability({"q": {"type": "noul", "noul": 0.91}}, "q"), 0.91)
        self.assertEqual(Questions.noul_probability({"q": {"type": "noul", "noul": 0}}, "q"), 0.0)
        self.assertEqual(Questions.noul_probability({"q": {"type": "noul", "noul": 1.4}}, "q"), 1.0)
        self.assertIsNone(Questions.noul_probability({"q": {"type": "noul", "noul": "yes"}}, "q"))
        self.assertIsNone(Questions.noul_probability({"q": {"type": "noul", "noul": True}}, "q"))

    def test_choice_answers(self):
        answers = {"t": {"type": "choice", "choice": "billing",
                         "probabilities": {"billing": 0.88, "sales": 0.12}, "confidence": 0.76}}
        self.assertEqual(Questions.choice_label(answers, "t"), "billing")
        self.assertEqual(Questions.choice_probabilities(answers, "t"), {"billing": 0.88, "sales": 0.12})
        self.assertEqual(Questions.confidence(answers, "t"), 0.76)

    def test_a_label_outside_the_offered_options_is_rejected(self):
        answers = {"t": {"type": "choice", "choice": "refunds", "probabilities": {"billing": 1.0}}}
        self.assertIsNone(Questions.choice_label(answers, "t"))

    def test_score_and_confidence(self):
        answers = {"s": {"type": "score", "score": 1.5, "confidence": 0.4}}
        self.assertEqual(Questions.score_value(answers, "s"), 1.5)
        self.assertEqual(Questions.confidence(answers, "s"), 0.4)
        self.assertIsNone(Questions.score_value({"s": {"type": "noul", "noul": 1.0}}, "s"))


class TestEndpoints(unittest.TestCase):
    def test_systemone_paths(self):
        self.assertEqual(SystemOneClient.build_endpoint("systemone", "https://api.typesafe.ai"),
                         "https://api.typesafe.ai/v1/systemone")
        # The TypeSafe SDK appends /v1/systemone to the OpenRouter base.
        self.assertEqual(SystemOneClient.build_endpoint("systemone", "https://openrouter.ai/api"),
                         "https://openrouter.ai/api/v1/systemone")

    def test_decisions_path(self):
        self.assertEqual(SystemOneClient.build_endpoint("decisions", "https://openrouter.ai/api"),
                         "https://openrouter.ai/api/alpha/decisions")

    def test_a_pasted_full_endpoint_is_not_doubled(self):
        self.assertEqual(SystemOneClient.build_endpoint("systemone", "https://api.typesafe.ai/v1/systemone"),
                         "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(SystemOneClient.build_endpoint("decisions", "https://openrouter.ai/api/alpha/decisions"),
                         "https://openrouter.ai/api/alpha/decisions")

    def test_trailing_slash_and_empty_base(self):
        self.assertEqual(SystemOneClient.build_endpoint("systemone", "https://api.typesafe.ai/"),
                         "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(SystemOneClient.build_endpoint("systemone", ""),
                         "https://api.typesafe.ai/v1/systemone")

    def test_cloudflare_needs_an_account_and_ignores_the_base_path(self):
        self.assertEqual(
            SystemOneClient.build_endpoint("cloudflare", "", "acct123"),
            "https://api.cloudflare.com/client/v4/accounts/acct123/ai/run/@cf/cloudflare/clef")
        with self.assertRaises(DecisionError):
            SystemOneClient.build_endpoint("cloudflare", "", "")

    def test_unknown_shape_is_rejected(self):
        with self.assertRaises(DecisionError):
            SystemOneClient.build_endpoint("openai", "https://x")

    def test_cloudflare_model_name_is_validated_early(self):
        with self.assertRaises(DecisionError):
            make_client(shape="cloudflare", account_id="a", model="gpt-5")
        make_client(shape="cloudflare", account_id="a", model="clef-flash")


class TestClient(unittest.TestCase):
    def test_request_carries_the_documented_body_and_bearer_token(self):
        with mock.patch.object(SystemOneClient.urlrequest, PATCH,
                               return_value=FakeResponse(answers_body(**{"q": {"type": "noul", "noul": 0.9}}))) as post:
            make_client().evaluate("some state", {"q": Questions.noul("Is it?")})
        request = post.call_args[0][0]
        self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertEqual(sent_body(request),
                         {"model": "jev-1.13", "state": "some state",
                          "questions": {"q": {"type": "noul", "instructions": "Is it?"}}})

    def test_a_non_ascii_state_is_sent_as_utf8_json(self):
        with mock.patch.object(SystemOneClient.urlrequest, PATCH,
                               return_value=FakeResponse(answers_body(**{"q": {"type": "noul", "noul": 0.1}}))) as post:
            make_client().evaluate("\u4f60\u597d", {"q": Questions.noul("ok?")})
        self.assertEqual(sent_body(post.call_args[0][0])["state"], "\u4f60\u597d")

    def test_http_error_raises_with_the_body_for_diagnosis(self):
        error = HTTPError("https://x", 400, "Bad Request", {},
                          __import__("io").BytesIO(b"bad schema"))
        with mock.patch.object(SystemOneClient.urlrequest, PATCH, side_effect=error):
            with self.assertRaises(DecisionError) as caught:
                make_client().evaluate("s", {"q": Questions.noul("ok?")})
        self.assertIn("400", str(caught.exception))
        self.assertIn("bad schema", str(caught.exception))

    def test_transport_failure_and_bad_json_raise_decision_error(self):
        with mock.patch.object(SystemOneClient.urlrequest, PATCH, side_effect=URLError("boom")):
            with self.assertRaises(DecisionError):
                make_client().evaluate("s", {"q": Questions.noul("ok?")})
        with mock.patch.object(SystemOneClient.urlrequest, PATCH,
                               return_value=FakeResponse(raw=b"<html>not json</html>")):
            with self.assertRaises(DecisionError):
                make_client().evaluate("s", {"q": Questions.noul("ok?")})

    def test_a_response_without_answers_is_rejected(self):
        with mock.patch.object(SystemOneClient.urlrequest, PATCH,
                               return_value=FakeResponse({"model": "x"})):
            with self.assertRaises(DecisionError):
                make_client().evaluate("s", {"q": Questions.noul("ok?")})

    def test_missing_credentials_fail_before_a_request_is_made(self):
        with mock.patch.object(SystemOneClient.urlrequest, PATCH) as post:
            with self.assertRaises(DecisionError):
                make_client(api_key="").evaluate("s", {"q": Questions.noul("ok?")})
        post.assert_not_called()

    def test_usage_reads_cost_only_when_the_provider_reports_it(self):
        with_cost = SystemOneClient.usage_of({"usage": {"input_tokens": 275, "output_tokens": 20, "cost": 0.00003}})
        self.assertEqual(with_cost, {"input_tokens": 275, "output_tokens": 20, "cost": 0.00003})
        self.assertEqual(SystemOneClient.usage_of({"usage": {"input_tokens": 5}}),
                         {"input_tokens": 5, "output_tokens": 0})
        self.assertEqual(SystemOneClient.usage_of({}), {"input_tokens": 0, "output_tokens": 0})
        # The reference client reads a top-level cost too, so accept both spellings.
        self.assertEqual(SystemOneClient.usage_of({"usage": {"input_tokens": 1}, "cost": 0.5}),
                         {"input_tokens": 1, "output_tokens": 0, "cost": 0.5})


class TestEngineFailOpen(unittest.TestCase):
    def test_ask_returns_none_instead_of_raising(self):
        engine = DecisionEngine.DecisionEngine(make_client())
        with mock.patch.object(SystemOneClient.urlrequest, PATCH, side_effect=URLError("down")):
            answers = engine.ask("s", {"q": Questions.noul("ok?")}, purpose="test")
        self.assertIsNone(answers)
        self.assertEqual(engine.failures, 1)
        self.assertEqual(engine.calls, 1)
        self.assertIn("down", engine.last_error)

    def test_from_config_returns_none_without_a_decision_platform(self):
        self.assertIsNone(DecisionEngine.DecisionEngine.from_config({"platforms": {}}))
        self.assertIsNone(DecisionEngine.DecisionEngine.from_config({}))
        self.assertIsNone(DecisionEngine.DecisionEngine.from_config(None))

    def test_from_config_finds_the_decision_platform_and_maps_the_shape(self):
        config = {"platforms": {
            "openai_1": {"group": "online", "api_key": "k"},
            "jev_2": {"group": "decision", "api_key": "k", "model": "jev-1.13",
                      "api_url": "https://openrouter.ai/api", "api_format": "SystemOne"},
        }}
        engine = DecisionEngine.DecisionEngine.from_config(config)
        self.assertIsNotNone(engine)
        self.assertEqual(engine.client.endpoint, "https://openrouter.ai/api/v1/systemone")

    def test_shape_aliases(self):
        self.assertEqual(DecisionEngine.resolve_shape("Cloudflare"), SystemOneClient.CLOUDFLARE)
        self.assertEqual(DecisionEngine.resolve_shape("Decisions"), SystemOneClient.DECISIONS)
        self.assertEqual(DecisionEngine.resolve_shape(""), SystemOneClient.SYSTEMONE)
        self.assertEqual(DecisionEngine.resolve_shape("nonsense"), SystemOneClient.SYSTEMONE)


class TestEngineBatching(unittest.TestCase):
    @staticmethod
    def _build(index, item):
        return ({"source": item, "translated": "T:" + item},
                {"i%d_lang" % index: Questions.noul("Is items.%d.translated in the target language?" % index),
                 "i%d_res" % index: Questions.noul("Does items.%d.translated keep text from items.%d.source?" % (index, index))})

    @staticmethod
    def _reply(request):
        body = sent_body(request)
        return FakeResponse(answers_body(**{qid: {"type": "noul", "noul": 0.9} for qid in body["questions"]}))

    def test_many_items_are_split_and_every_item_is_answered(self):
        engine = DecisionEngine.DecisionEngine(make_client())
        items = ["line%d" % i for i in range(100)]
        sent = []

        def fake_urlopen(request, timeout=None):
            sent.append(sent_body(request))
            return self._reply(request)

        with mock.patch.object(SystemOneClient.urlrequest, PATCH, side_effect=fake_urlopen):
            results = engine.decide_batch(items, self._build, purpose="test")

        self.assertEqual(len(results), 100)
        self.assertEqual(engine.calls, len(sent))
        self.assertGreater(len(sent), 1, "100 items at 2 questions each must not fit in one request")
        for body in sent:
            self.assertLessEqual(len(body["questions"]), Questions.MAX_QUESTIONS)
        for index in range(100):
            self.assertEqual(Questions.noul_probability(results[index], "i%d_lang" % index), 0.9)
            self.assertEqual(Questions.noul_probability(results[index], "i%d_res" % index), 0.9)

    def test_state_indices_stay_global_across_chunks(self):
        """A local index would silently point every question at the wrong line."""
        engine = DecisionEngine.DecisionEngine(make_client())
        items = ["line%d" % i for i in range(100)]
        seen = []

        def fake_urlopen(request, timeout=None):
            seen.append(sorted(sent_body(request)["state"]["items"], key=int))
            return self._reply(request)

        with mock.patch.object(SystemOneClient.urlrequest, PATCH, side_effect=fake_urlopen):
            engine.decide_batch(items, self._build)

        all_indices = [int(i) for chunk in seen for i in chunk]
        self.assertEqual(sorted(all_indices), list(range(100)))
        # The second request continues at 32, it does not restart at 0.
        self.assertEqual(seen[1][0], "32")

    def test_a_failed_chunk_leaves_unknown_not_false(self):
        engine = DecisionEngine.DecisionEngine(make_client())
        items = ["line%d" % i for i in range(64)]
        calls = {"n": 0}

        def flaky(request, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise URLError("down")
            return self._reply(request)

        with mock.patch.object(SystemOneClient.urlrequest, PATCH, side_effect=flaky):
            results = engine.decide_batch(items, self._build)

        # First chunk unknown, later chunks answered: fail open, do not cascade.
        self.assertIsNone(Questions.noul_probability(results[0], "i0_lang"))
        self.assertEqual(Questions.noul_probability(results[-1], "i63_lang"), 0.9)
        self.assertEqual(engine.failures, 1)

    def test_state_budget_splits_earlier_than_the_question_cap(self):
        engine = DecisionEngine.DecisionEngine(make_client())
        items = ["x" * 500 for _ in range(20)]
        sent = []

        def fake_urlopen(request, timeout=None):
            sent.append(len(sent_body(request)["state"]["items"]))
            return self._reply(request)

        with mock.patch.object(SystemOneClient.urlrequest, PATCH, side_effect=fake_urlopen):
            engine.decide_batch(items, self._build, max_state_chars=2000)
        self.assertGreater(len(sent), 1)
        self.assertTrue(all(count < 20 for count in sent))

    def test_empty_batch_makes_no_request(self):
        engine = DecisionEngine.DecisionEngine(make_client())
        with mock.patch.object(SystemOneClient.urlrequest, PATCH) as post:
            self.assertEqual(engine.decide_batch([], self._build), [])
        post.assert_not_called()

    def test_usage_is_accumulated_and_summarised(self):
        engine = DecisionEngine.DecisionEngine(make_client())
        body = {"answers": {"q": {"type": "noul", "noul": 1.0}},
                "usage": {"input_tokens": 100, "output_tokens": 5, "cost": 0.002}}
        with mock.patch.object(SystemOneClient.urlrequest, PATCH, return_value=FakeResponse(body)):
            engine.ask("s", {"q": Questions.noul("ok?")})
        summary = engine.summary()
        self.assertEqual(summary["input_tokens"], 100)
        self.assertEqual(summary["output_tokens"], 5)
        self.assertEqual(summary["cost"], 0.002)
        self.assertEqual(summary["failures"], 0)


if __name__ == "__main__":
    unittest.main()