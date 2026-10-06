"""The term consistency sweep: alias vs derived name, and unsafe glossary keys.

The behaviour under test is the one the naive substring merge gets wrong. Given

    风神 (Wind God, a character)  /  风神斩 (Wind God Slash, a skill)
    亚瑟王 (King Arthur)         /  亚瑟王之剑 (King Arthur's Sword)

the short term is NOT a variant of the long one. Merging them loses the standalone entry, so
every later mention of 风神 alone is translated ad hoc - the drift this sweep exists to stop.

The replies below are keyed by the SOURCE PAIR rather than by request index, so the tests also
assert how the pair is presented: the shorter source must always be the "a" side, because the
questions are phrased against that.
"""
import unittest
from unittest import mock
from urllib.error import URLError

from ModuleFolders.Infrastructure.DecisionEngine import ConsistencySweep as sweep_module
from ModuleFolders.Infrastructure.DecisionEngine import DecisionEngine, Questions, SystemOneClient
from ModuleFolders.Infrastructure.DecisionEngine.ConsistencySweep import ConsistencySweep
from ModuleFolders.Infrastructure.DecisionEngine.SystemOneClient import SystemOneClient as Client
from tests.test_decision_engine import FakeResponse, sent_body

FENGSHEN = "\u98ce\u795e"
FENGSHEN_ZHAN = "\u98ce\u795e\u65a9"
ARTHUR = "\u4e9a\u745f\u738b"
EXCALIBUR = "\u4e9a\u745f\u738b\u4e4b\u5251"


def make_engine():
    return DecisionEngine.DecisionEngine(Client(api_key="k", model="jev-latest"))


def replies(relations=None, generic=None, consistency=None, calls=None):
    """Answer by source (pair), never by index, so index assumptions cannot hide a bug."""
    def fake(request, timeout=None):
        body = sent_body(request)
        if calls is not None:
            calls.append(body)
        answers = {}
        for qid in body["questions"]:
            kind, index = qid[0], int(qid[1:])
            item = body["state"]["items"][str(index)]
            if kind == "r":
                key = (item["a"]["source"], item["b"]["source"])
                label = (relations or {}).get(key)
                if label is None:
                    continue
                answers[qid] = {"type": "choice", "choice": label,
                                "probabilities": {label: 0.9}, "confidence": 0.9}
            elif kind == "g":
                probability = (generic or {}).get(item["source"])
                if probability is None:
                    continue
                answers[qid] = {"type": "noul", "noul": probability}
            elif kind == "c":
                key = (item["a"]["source"], item["b"]["source"])
                probability = (consistency or {}).get(key)
                if probability is None:
                    continue
                answers[qid] = {"type": "noul", "noul": probability}
        return FakeResponse({"answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}})
    return fake


def run_sources(terms, **kwargs):
    sweep = ConsistencySweep(make_engine())
    with mock.patch.object(SystemOneClient.urlrequest, "urlopen", side_effect=replies(**kwargs)):
        return sweep.sweep_sources(terms)


def term(source, kind="term"):
    return {"source": source, "kind": kind}


class TestGlossaryKeySafety(unittest.TestCase):
    def test_single_character_sources_are_dropped_and_never_sent(self):
        calls = []
        outcome = run_sources(
            [term("\u5251"), term("\u738b"), term(FENGSHEN)],  # 剑, 王
            generic={FENGSHEN: 0.01},
            calls=calls,
        )
        self.assertEqual(sorted(outcome.single_character), sorted(["\u5251", "\u738b"]))
        self.assertNotIn(FENGSHEN, outcome.dropped)
        self.assertTrue(calls, "the surviving term is still judged")
        # The dropped characters never reach a request, not even as context.
        for body in calls:
            for item in body["state"]["items"].values():
                for side in ("a", "b"):
                    if side in item:
                        self.assertNotIn(item[side]["source"], ["\u5251", "\u738b"])
                self.assertNotIn(item.get("source"), ["\u5251", "\u738b"])

    def test_a_term_judged_too_generic_is_reported_for_removal(self):
        outcome = run_sources(
            [term("\u5927\u4eba"), term(FENGSHEN)],  # 大人 is an everyday honorific
            generic={"\u5927\u4eba": 0.93, FENGSHEN: 0.02},
        )
        self.assertEqual(outcome.generic, ["\u5927\u4eba"])
        self.assertIn("\u5927\u4eba", outcome.dropped)
        self.assertNotIn(FENGSHEN, outcome.dropped)

    def test_a_dropped_term_is_never_paired_for_a_relation_question(self):
        """Ordering matters: drop first, then pair, so junk cannot reach a merge decision."""
        calls = []
        run_sources(
            [term("\u5251"), term(FENGSHEN_ZHAN), term(FENGSHEN)],
            relations={(FENGSHEN, FENGSHEN_ZHAN): sweep_module.SUBORDINATE},
            calls=calls,
        )
        for body in calls:
            for item in body["state"]["items"].values():
                for side in ("a", "b"):
                    if side in item:
                        self.assertNotEqual(item[side]["source"], "\u5251")


class TestRelationVerdicts(unittest.TestCase):
    def test_a_derived_name_stays_a_separate_entry(self):
        outcome = run_sources(
            [term(FENGSHEN, "character"), term(FENGSHEN_ZHAN)],
            relations={(FENGSHEN, FENGSHEN_ZHAN): sweep_module.SUBORDINATE},
        )
        self.assertIn((FENGSHEN, FENGSHEN_ZHAN), outcome.keep_separate)
        self.assertEqual(outcome.subordinate, [(FENGSHEN, FENGSHEN_ZHAN)])
        self.assertEqual(outcome.relations[(FENGSHEN, FENGSHEN_ZHAN)], sweep_module.SUBORDINATE)

    def test_the_sword_and_the_king_are_two_entries(self):
        outcome = run_sources(
            [term(ARTHUR, "character"), term(EXCALIBUR)],
            relations={(ARTHUR, EXCALIBUR): sweep_module.SUBORDINATE},
        )
        self.assertIn((ARTHUR, EXCALIBUR), outcome.keep_separate)

    def test_an_alias_is_left_mergeable(self):
        """A genuine short form must keep behaving as it did before the sweep existed."""
        outcome = run_sources(
            [term(ARTHUR, "character"), term(ARTHUR + "\u965b\u4e0b")],  # 亚瑟王陛下
            relations={(ARTHUR, ARTHUR + "\u965b\u4e0b"): sweep_module.ALIAS},
        )
        self.assertEqual(outcome.keep_separate, set())
        self.assertEqual(outcome.subordinate, [])
        self.assertEqual(outcome.relations[(ARTHUR, ARTHUR + "\u965b\u4e0b")], sweep_module.ALIAS)

    def test_unrelated_overlap_is_separated_but_not_consistency_checked(self):
        outcome = run_sources(
            [term("\u5929\u4e0b"), term("\u5929\u4e0b\u65e0\u53cc")],  # coincidental overlap
            relations={("\u5929\u4e0b", "\u5929\u4e0b\u65e0\u53cc"): sweep_module.UNRELATED},
        )
        self.assertIn(("\u5929\u4e0b", "\u5929\u4e0b\u65e0\u53cc"), outcome.keep_separate)
        self.assertEqual(outcome.subordinate, [], "no point checking translations that share no name")

    def test_an_unreadable_relation_keeps_the_old_behaviour(self):
        """Fail open: no verdict means the pair is merged exactly as before."""
        outcome = run_sources([term(ARTHUR), term(EXCALIBUR)], relations={})
        self.assertEqual(outcome.keep_separate, set())
        self.assertEqual(outcome.pairs_total, 1)
        self.assertEqual(outcome.relations, {})

    def test_a_coin_flip_choice_is_treated_as_unknown(self):
        def flat(request, timeout=None):
            body = sent_body(request)
            answers = {qid: {"type": "choice", "choice": "subordinate",
                             "probabilities": {"alias": 0.34, "subordinate": 0.33, "unrelated": 0.33}}
                       for qid in body["questions"] if qid.startswith("r")}
            return FakeResponse({"answers": answers})

        sweep = ConsistencySweep(make_engine())
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen", side_effect=flat):
            outcome = sweep.sweep_sources([term(ARTHUR), term(EXCALIBUR)])
        self.assertEqual(outcome.keep_separate, set())

    def test_a_provider_outage_changes_nothing_but_the_single_character_rule(self):
        sweep = ConsistencySweep(make_engine())
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen", side_effect=URLError("down")):
            outcome = sweep.sweep_sources([term("\u5251"), term(ARTHUR), term(EXCALIBUR)])
        self.assertEqual(outcome.single_character, ["\u5251"])
        self.assertEqual(outcome.keep_separate, set())
        self.assertEqual(outcome.generic, [])


class TestPairFinding(unittest.TestCase):
    def test_containment_is_found_in_one_direction_only(self):
        pairs, dropped = sweep_module.find_containment_pairs([FENGSHEN, FENGSHEN_ZHAN])
        self.assertEqual(pairs, [(0, 1)])
        self.assertEqual(dropped, 0)

    def test_unrelated_sources_produce_no_pairs(self):
        self.assertEqual(sweep_module.find_containment_pairs([ARTHUR, FENGSHEN]), ([], 0))

    def test_equal_sources_are_not_a_pair(self):
        self.assertEqual(sweep_module.find_containment_pairs([ARTHUR, ARTHUR]), ([], 0))

    def test_pairs_are_ordered_longest_source_first(self):
        pairs, _ = sweep_module.find_containment_pairs([FENGSHEN, FENGSHEN_ZHAN, ARTHUR, EXCALIBUR])
        self.assertEqual(pairs, [(2, 3), (0, 1)])

    def test_the_cap_reports_what_it_left_unjudged(self):
        sources = [ARTHUR, ARTHUR + "\u4e4b\u5251", ARTHUR + "\u4e4b\u5251\u4e4b\u5f71"]
        pairs, dropped = sweep_module.find_containment_pairs(sources, max_pairs=2)
        self.assertEqual(len(pairs), 2)
        self.assertGreater(dropped, 0)

    def test_empty_sources_are_ignored(self):
        self.assertEqual(sweep_module.find_containment_pairs(["", ARTHUR]), ([], 0))


class TestRequestShape(unittest.TestCase):
    def test_the_relation_question_offers_the_three_relations(self):
        calls = []
        run_sources([term(FENGSHEN, "character"), term(FENGSHEN_ZHAN, "term")],
                    relations={(FENGSHEN, FENGSHEN_ZHAN): sweep_module.SUBORDINATE}, calls=calls)
        relation = next(q for body in calls for qid, q in body["questions"].items() if qid.startswith("r"))
        self.assertEqual(relation["type"], "choice")
        self.assertEqual(sorted(relation["criteria"]), [sweep_module.ALIAS, sweep_module.SUBORDINATE,
                                                        sweep_module.UNRELATED])

    def test_the_pair_is_presented_short_side_first_with_both_kinds(self):
        calls = []
        run_sources([term(FENGSHEN, "character"), term(FENGSHEN_ZHAN, "term")],
                    relations={(FENGSHEN, FENGSHEN_ZHAN): sweep_module.SUBORDINATE}, calls=calls)
        item = next(v for body in calls for v in body["state"]["items"].values() if "a" in v)
        self.assertEqual(item["a"]["source"], FENGSHEN)
        self.assertEqual(item["b"]["source"], FENGSHEN_ZHAN)
        question = next(q for body in calls for qid, q in body["questions"].items() if qid.startswith("r"))
        self.assertIn("items.0.a.source", question["instructions"])
        self.assertIn("items.0.b.source", question["instructions"])

    def test_many_pairs_are_split_across_requests(self):
        base = "\u4e9a\u745f"
        terms = [term(base + "\u7b2c%d\u7ae0" % i) for i in range(40)] + [term(base)]
        calls = []
        run_sources(terms, relations={}, calls=calls)
        self.assertGreater(len(calls), 1)
        for body in calls:
            self.assertLessEqual(len(body["questions"]), Questions.MAX_QUESTIONS)


class TestTranslationConsistency(unittest.TestCase):
    ENTRIES = [
        {"source": FENGSHEN, "translation": "Wind God"},
        {"source": FENGSHEN_ZHAN, "translation": "Wind God Slash"},
        {"source": ARTHUR, "translation": "King Arthur"},
        {"source": EXCALIBUR, "translation": "Sword of the King"},
    ]
    PAIRS = [(FENGSHEN, FENGSHEN_ZHAN), (ARTHUR, EXCALIBUR)]

    def sweep(self, **kwargs):
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen", side_effect=replies(**kwargs)):
            return ConsistencySweep(make_engine()).sweep_translations(self.ENTRIES, self.PAIRS)

    def test_a_shared_name_translated_two_ways_is_reported(self):
        outcome = self.sweep(consistency={(FENGSHEN, FENGSHEN_ZHAN): 0.95,
                                          (ARTHUR, EXCALIBUR): 0.02})
        self.assertEqual(outcome.judged, 2)
        self.assertEqual(len(outcome.inconsistent), 1)
        short, long_source, probability = outcome.inconsistent[0]
        self.assertEqual((short, long_source), (ARTHUR, EXCALIBUR))
        self.assertAlmostEqual(probability, 0.02)

    def test_consistent_translations_are_not_reported(self):
        outcome = self.sweep(consistency={(FENGSHEN, FENGSHEN_ZHAN): 0.97,
                                          (ARTHUR, EXCALIBUR): 0.88})
        self.assertEqual(outcome.inconsistent, [])
        self.assertEqual(outcome.judged, 2)

    def test_a_pair_missing_a_translation_is_skipped_not_flagged(self):
        """Nothing to compare is not the same as inconsistent."""
        entries = [{"source": ARTHUR, "translation": "King Arthur"}]
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen") as post:
            outcome = ConsistencySweep(make_engine()).sweep_translations(entries, self.PAIRS)
        self.assertEqual(outcome.inconsistent, [])
        self.assertEqual(outcome.skipped, 2)
        post.assert_not_called()

    def test_no_pairs_means_no_request(self):
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen") as post:
            outcome = ConsistencySweep(make_engine()).sweep_translations(self.ENTRIES, [])
        self.assertEqual(outcome.judged, 0)
        post.assert_not_called()

    def test_only_the_pairs_handed_in_are_asked_about(self):
        calls = []
        with mock.patch.object(SystemOneClient.urlrequest, "urlopen",
                               side_effect=replies(consistency={}, calls=calls)):
            ConsistencySweep(make_engine()).sweep_translations(self.ENTRIES, [(FENGSHEN, FENGSHEN_ZHAN)])
        asked = [qid for body in calls for qid in body["questions"]]
        self.assertEqual(asked, ["c0"])
        state = calls[0]["state"]["items"]["0"]
        self.assertEqual(state["a"]["source"], FENGSHEN)
        self.assertEqual(state["a"]["translation"], "Wind God")
        self.assertEqual(state["b"]["translation"], "Wind God Slash")




def grouped(sources, keep_separate=None, enable_short_name_merge=True):
    raw = {
        source: {"source": source, "merged_sources": [source],
                 "candidates": [{"candidate_source": source, "type": "term"}]}
        for source in sources
    }
    return sweep_module.group_sources(raw, keep_separate, enable_short_name_merge)


class TestGrouping(unittest.TestCase):
    """The rule the sweep exists to correct, tested without Qt or a network."""

    TWO_PAIRS = [(FENGSHEN, FENGSHEN_ZHAN), (ARTHUR, EXCALIBUR)]

    def test_a_derived_name_is_not_swallowed(self):
        groups, aliases = grouped([FENGSHEN, FENGSHEN_ZHAN, ARTHUR, EXCALIBUR], self.TWO_PAIRS)
        self.assertEqual(sorted(groups), sorted([FENGSHEN, FENGSHEN_ZHAN, ARTHUR, EXCALIBUR]))
        for source in (FENGSHEN, ARTHUR):
            self.assertEqual(aliases[source], source)

    def test_without_a_verdict_the_old_merge_still_happens(self):
        """Fail open: no decision model means today's behaviour, unchanged."""
        groups, aliases = grouped([FENGSHEN, FENGSHEN_ZHAN, ARTHUR, EXCALIBUR])
        self.assertEqual(sorted(groups), sorted([FENGSHEN_ZHAN, EXCALIBUR]))
        self.assertEqual(aliases[FENGSHEN], FENGSHEN_ZHAN)
        self.assertEqual(aliases[ARTHUR], EXCALIBUR)
        self.assertEqual(sorted(groups[EXCALIBUR]["merged_sources"]), sorted([ARTHUR, EXCALIBUR]))

    def test_an_alias_still_merges(self):
        groups, aliases = grouped([ARTHUR, ARTHUR + "\u965b\u4e0b"], keep_separate=set())
        self.assertEqual(list(groups), [ARTHUR + "\u965b\u4e0b"])
        self.assertEqual(aliases[ARTHUR], ARTHUR + "\u965b\u4e0b")

    def test_a_short_form_joins_its_own_holder_not_the_longest_one(self):
        """亚瑟 is a short form of 亚瑟王, and only the sword pair is derived.

        Every containment pair is judged, so 亚瑟 is also blocked from the sword and then
        falls through to the name it actually abbreviates.
        """
        short = ARTHUR[:2]  # 亚瑟
        groups, aliases = grouped(
            [short, ARTHUR, EXCALIBUR],
            keep_separate={(ARTHUR, EXCALIBUR), (short, EXCALIBUR)},
        )
        self.assertEqual(sorted(groups), sorted([ARTHUR, EXCALIBUR]))
        self.assertEqual(aliases[short], ARTHUR)
        self.assertEqual(groups[ARTHUR]["merged_sources"], [ARTHUR, short])

    def test_merged_candidates_come_from_every_member(self):
        groups, _ = grouped([ARTHUR, ARTHUR + "\u965b\u4e0b"])
        merged = list(groups.values())[0]
        self.assertEqual(len(merged["candidates"]), 2)

    def test_the_merge_can_be_switched_off_entirely(self):
        groups, aliases = grouped([ARTHUR, EXCALIBUR], enable_short_name_merge=False)
        self.assertEqual(sorted(groups), sorted([ARTHUR, EXCALIBUR]))
        self.assertEqual(aliases[ARTHUR], ARTHUR)

    def test_a_reversed_pair_does_not_block_the_merge(self):
        """The tuple is ordered (shorter, longer); a reversed one must not silently pass."""
        groups, _ = grouped([FENGSHEN, FENGSHEN_ZHAN], keep_separate={(FENGSHEN_ZHAN, FENGSHEN)})
        self.assertEqual(list(groups), [FENGSHEN_ZHAN])

    def test_an_empty_input_is_fine(self):
        self.assertEqual(grouped([]), ({}, {}))


class TestSweepFeedsGrouping(unittest.TestCase):
    """The two halves joined: a verdict from the model must reach the grouping rule."""

    def test_the_reported_pairs_are_exactly_what_grouping_needs(self):
        terms = [term(FENGSHEN, "character"), term(FENGSHEN_ZHAN, "term"),
                 term(ARTHUR, "character"), term(EXCALIBUR, "term")]
        outcome = run_sources(
            terms,
            relations={(FENGSHEN, FENGSHEN_ZHAN): sweep_module.SUBORDINATE,
                       (ARTHUR, EXCALIBUR): sweep_module.SUBORDINATE},
        )
        raw = {t["source"]: {"source": t["source"], "candidates": []} for t in terms}
        groups, _ = sweep_module.group_sources(raw, outcome.keep_separate)
        self.assertEqual(len(groups), 4, "each derived name keeps its own entry")




def original_group_sources(raw_grouped_inputs, enable_short_name_merge=True):
    """The grouping exactly as it stood before the sweep, copied from the previous commit.

    Kept here as a characterisation reference: with no verdict from the model, group_sources
    must reproduce this byte for byte, so enabling the decision layer cannot change extraction
    for anyone who has not configured it.
    """
    if not enable_short_name_merge:
        grouped_inputs = {
            source: {
                "source": source,
                "merged_sources": [source],
                "candidates": list(grouped_item.get("candidates", [])),
            }
            for source, grouped_item in raw_grouped_inputs.items()
        }
        source_aliases = {source: source for source in grouped_inputs}
    else:
        sorted_sources = sorted(raw_grouped_inputs.keys(), key=lambda s: (-len(s), s))
        grouped_inputs, source_aliases, consumed_sources = {}, {}, set()

        for source in sorted_sources:
            if source in consumed_sources:
                continue

            merged_group = {
                "source": source,
                "merged_sources": [source],
                "candidates": list(raw_grouped_inputs[source].get("candidates", [])),
            }
            grouped_inputs[source] = merged_group
            source_aliases[source] = source
            consumed_sources.add(source)

            for other_source in sorted_sources:
                if other_source in consumed_sources or other_source == source:
                    continue
                if other_source in source:  # short source attaches to the long one
                    merged_group["merged_sources"].append(other_source)
                    merged_group["candidates"].extend(
                        raw_grouped_inputs[other_source].get("candidates", [])
                    )
                    source_aliases[other_source] = source
                    consumed_sources.add(other_source)
    return grouped_inputs, source_aliases


class TestParityWithTheOriginalRule(unittest.TestCase):
    """No decision model must mean no change at all - asserted, not assumed."""

    CASES = [
        [],
        [ARTHUR],
        [ARTHUR, EXCALIBUR],
        [FENGSHEN, FENGSHEN_ZHAN],
        [ARTHUR, ARTHUR + "\u965b\u4e0b", EXCALIBUR],                   # alias chain
        [ARTHUR[:2], ARTHUR, EXCALIBUR],                               # three-level nest
        [FENGSHEN, FENGSHEN_ZHAN, ARTHUR, EXCALIBUR],                  # two families
        ["ab", "abcd", "bcde"],                                        # overlapping, not nested
        [ARTHUR, ARTHUR],                                              # duplicate key collapses
    ]

    def raw(self, sources):
        return {
            source: {"source": source, "merged_sources": [source],
                     "candidates": [{"candidate_source": source, "type": "term"}]}
            for source in sources
        }

    def test_an_empty_verdict_set_reproduces_the_original_exactly(self):
        for sources in self.CASES:
            raw = self.raw(sources)
            with self.subTest(sources=sources):
                self.assertEqual(sweep_module.group_sources(raw), original_group_sources(raw))

    def test_an_empty_verdict_set_reproduces_the_original_with_merging_off(self):
        for sources in self.CASES:
            raw = self.raw(sources)
            with self.subTest(sources=sources):
                self.assertEqual(
                    sweep_module.group_sources(raw, enable_short_name_merge=False),
                    original_group_sources(raw, enable_short_name_merge=False),
                )

    def test_the_one_divergence_is_the_derived_term_gaining_its_own_entry(self):
        """Blocking one pair must change that pair's membership and nothing else.

        The group KEY survives either way - what changes is that the short term stops being
        absorbed, so it becomes an entry of its own instead of vanishing.
        """
        sources = [FENGSHEN, FENGSHEN_ZHAN, ARTHUR, EXCALIBUR]
        raw = self.raw(sources)
        original, original_aliases = original_group_sources(raw)
        swept, swept_aliases = sweep_module.group_sources(raw, {(FENGSHEN, FENGSHEN_ZHAN)})

        # Before: the short term was absorbed and had no entry of its own.
        self.assertEqual(original[FENGSHEN_ZHAN]["merged_sources"], [FENGSHEN_ZHAN, FENGSHEN])
        self.assertEqual(original_aliases[FENGSHEN], FENGSHEN_ZHAN)
        self.assertNotIn(FENGSHEN, original)

        # After: it keeps its own group, and nothing about the other family changed.
        self.assertEqual(swept[FENGSHEN_ZHAN]["merged_sources"], [FENGSHEN_ZHAN])
        self.assertEqual(swept[FENGSHEN]["merged_sources"], [FENGSHEN])
        self.assertEqual(swept_aliases[FENGSHEN], FENGSHEN)
        # The other family is untouched, including the member it absorbed (亚瑟王).
        self.assertEqual(swept[EXCALIBUR], original[EXCALIBUR])
        self.assertEqual(original[EXCALIBUR]["merged_sources"], [EXCALIBUR, ARTHUR])
        self.assertEqual(set(swept) - set(original), {FENGSHEN})


if __name__ == "__main__":
    unittest.main()