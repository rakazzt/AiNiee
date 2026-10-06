"""Consistency sweep over extracted glossary terms, driven by typed decisions.

WHY THIS EXISTS
AiNiee groups extracted terms by a naive rule: any source that is a plain substring of a
longer source is merged into it. That is right for aliases and wrong for derived names.

    风神   (Wind God, a character)   and  风神斩 (Wind God Slash, a skill)
    亚瑟王 (King Arthur)             and  亚瑟王之剑 (King Arthur's Sword)

Neither short term is a variant of the long one - they are different entities, and the short
one appears on its own throughout the book. Merging them collapses two glossary entries into
one, so every standalone occurrence of 风神 loses its entry and gets translated ad hoc. That
is the retranslation drift this sweep prevents: the model decides which pairs are the same
entity (alias, short form, name with a title) and which are merely nested.

It also refuses glossary entries that are too generic to be safe. A single character matches
inside unrelated words, so forcing a fixed translation for it corrupts text elsewhere; those
are dropped without spending a request, and longer but still generic sources are judged.

Everything here is pure and Qt-free: pair finding, question building and verdict reading take
plain dicts, so the logic is testable without the app or the network.
"""

from ModuleFolders.Infrastructure.DecisionEngine import DecisionSettings, Questions

# Relations. Only an alias may be merged; the other two must stay separate entries.
ALIAS = "alias"
SUBORDINATE = "subordinate"
UNRELATED = "unrelated"
RELATIONS = (ALIAS, SUBORDINATE, UNRELATED)

# A one-character source is never a safe glossary key: it matches inside unrelated words, and
# a fixed translation for it corrupts text everywhere else. Dropped on length alone, so this
# costs nothing and behaves identically with or without a decision model.
MIN_SOURCE_LENGTH = 2

# A noul answer is a probability; these are the cuts. Below the floor a reading is treated as
# unknown, and every unknown falls back to what the extractor did before this sweep existed.
DECISION_THRESHOLD = 0.5

# Safety valve on request count. Containment pairs are far rarer than terms, so this normally
# never binds; when it does, the truncation is reported rather than silent.
MAX_PAIRS = 400


def is_single_character(source) -> bool:
    """True when a source is too short to be a safe glossary key."""
    return len(str(source or "").strip()) < MIN_SOURCE_LENGTH


def find_containment_pairs(sources, max_pairs: int = MAX_PAIRS):
    """Pairs (shorter, longer) where one source sits inside the other.

    Deterministic work stays in code: this is a string scan, not a judgment. Returns indices
    into sources, each pair once, ordered longest-source-first to match how the extractor
    walks them. Also returns how many pairs were dropped by the cap.
    """
    texts = [str(source or "").strip() for source in sources]
    pairs = []
    for long_index, long_text in enumerate(texts):
        if not long_text:
            continue
        for short_index, short_text in enumerate(texts):
            if short_index == long_index or not short_text:
                continue
            if len(short_text) >= len(long_text):
                continue  # the longer side is always the container
            if short_text in long_text:
                pairs.append((short_index, long_index))
    pairs.sort(key=lambda pair: (-len(texts[pair[1]]), -len(texts[pair[0]]), pair))
    dropped = max(0, len(pairs) - max_pairs)
    return pairs[:max_pairs], dropped


# --- questions ----------------------------------------------------------------
# Each pair travels as its own item, so both terms are always present in the request that
# asks about them and a reference such as items.3.a.source always resolves.

def relation_question(index: int) -> dict:
    """One question per pair: same entity, or one name built on the other?"""
    return {
        "r{0}".format(index): Questions.choice(
            ("What is the relationship between `items.{0}.a.source` and `items.{0}.b.source`? "
             "`items.{0}.a.source` is the shorter one.").format(index),
            {
                ALIAS: ("The same entity. The shorter is a short form, alias, nickname, or the "
                        "longer name with a title or honorific attached."),
                SUBORDINATE: ("Different entities, but the longer name is built on the shorter "
                              "one - an item, skill, technique, title or place named after it. "
                              "The shorter term is also used on its own."),
                UNRELATED: "The overlap is coincidental; the two names are not related.",
            },
        ),
    }


def generic_question(index: int) -> dict:
    """One question per term: is this too generic to be a safe glossary entry?"""
    return {
        "g{0}".format(index): Questions.noul(
            ("Is `items.{0}.source` too generic or too common to be a reliable glossary term? "
             "Answer yes for a single character, an everyday word, an honorific, or a fragment "
             "that would also match unrelated text - anything that would be wrong to force one "
             "fixed translation onto.").format(index),
            true_meaning="Too generic: forcing one translation would damage unrelated text.",
            false_meaning="Specific enough to be a real term in this work.",
        ),
    }


def consistency_question(index: int) -> dict:
    """One question per pair: is the shared part rendered the same way in both?"""
    return {
        "c{0}".format(index): Questions.noul(
            ("`items.{0}.b.source` is named after `items.{0}.a.source`. Does "
             "`items.{0}.b.translation` render that shared name exactly the way "
             "`items.{0}.a.translation` renders it? Answer no if the same name is translated "
             "two different ways.").format(index),
            true_meaning="The shared name is translated identically in both.",
            false_meaning="The same name is translated inconsistently between the two.",
        ),
    }


# --- reading verdicts ---------------------------------------------------------

def read_relation(answers, index: int, threshold: float = DECISION_THRESHOLD):
    """The chosen relation, or None when it could not be read (caller fails open)."""
    label = Questions.choice_label(answers, "r{0}".format(index))
    if label not in RELATIONS:
        return None
    probabilities = Questions.choice_probabilities(answers, "r{0}".format(index))
    if probabilities is not None and probabilities.get(label, 0.0) < threshold:
        # Too close to call: treat as unknown rather than acting on a coin flip.
        return None
    return label


def read_generic(answers, index: int):
    """Probability that a term is too generic, or None when unavailable."""
    return Questions.noul_probability(answers, "g{0}".format(index))


def read_consistency(answers, index: int):
    """Probability that the pair's translations are consistent, or None."""
    return Questions.noul_probability(answers, "c{0}".format(index))

# --- driving the decisions ----------------------------------------------------

class SourceSweep:
    """What the source-side sweep decided. Every field is data; nothing is mutated here."""

    def __init__(self, drop_single_character: bool = True, drop_generic: bool = True):
        self.single_character = []   # found by length alone, before any request
        self.generic = []            # judged too generic to be a safe glossary entry
        # What the policy actually removes. A finding the settings say to keep is still
        # reported, so "report only" is visible in the log rather than looking like a no-op.
        self.dropped = []
        self._drop_single_character = drop_single_character
        self._drop_generic = drop_generic
        self.relations = {}          # (shorter, longer) -> relation
        self.keep_separate = set()   # pairs that must NOT be merged into one entry
        self.subordinate = []        # (shorter, longer) pairs worth a consistency check
        self.pairs_total = 0
        self.pairs_dropped = 0       # pairs the cap left unjudged

    def record_single_character(self, source: str) -> None:
        self.single_character.append(source)
        if self._drop_single_character:
            self.dropped.append(source)

    def record_generic(self, source: str) -> None:
        self.generic.append(source)
        if self._drop_generic:
            self.dropped.append(source)

    def summary(self) -> dict:
        return {
            "single_character": len(self.single_character),
            "generic": len(self.generic),
            "dropped": len(self.dropped),
            "pairs": self.pairs_total,
            "pairs_judged": len(self.relations),
            "pairs_dropped": self.pairs_dropped,
            "kept_separate": len(self.keep_separate),
        }


class TranslationSweep:
    """Related pairs whose two translations disagree about the shared name."""

    def __init__(self):
        self.inconsistent = []   # (shorter, longer, probability_of_consistency)
        self.judged = 0
        self.skipped = 0

    def summary(self) -> dict:
        return {
            "judged": self.judged,
            "inconsistent": len(self.inconsistent),
            "skipped": self.skipped,
        }


class ConsistencySweep:
    """Runs the decisions and reports them. The caller applies the consequences."""

    def __init__(self, engine, max_pairs: int = MAX_PAIRS, settings=None):
        self.engine = engine
        # Settings decide what the sweep is allowed to do; max_pairs stays a constructor
        # argument so existing callers keep working without a settings map.
        self.settings = DecisionSettings.normalize(settings)
        self.max_pairs = max_pairs if settings is None else self.settings["max_pairs"]

    # --- source side: relations and junk terms --------------------------------

    def sweep_sources(self, terms) -> SourceSweep:
        """terms: [{"source": str, "kind": str}]. Returns decisions, changes nothing.

        Order matters and is deliberate: junk is removed first, so a dropped term is never
        merged into anything and never costs a relation question.
        """
        settings = self.settings
        outcome = SourceSweep(
            drop_single_character=settings["drop_single_character"],
            drop_generic=settings["drop_generic"],
        )
        kept = []
        for term in terms or []:
            source = str((term or {}).get("source", "")).strip()
            if not source:
                continue
            if is_single_character(source):
                outcome.record_single_character(source)
                if settings["drop_single_character"]:
                    continue  # removed by length, so no request is spent judging it
                # Keep it in play so the generic question can still have its say.
            kept.append({"source": source, "kind": str((term or {}).get("kind", "")).strip()})

        if kept and settings["generic_switch"]:
            answers = self.engine.decide_batch(kept, self._build_generic, purpose="term_generic")
            for index, term in enumerate(kept):
                probability = read_generic(answers[index], index)
                if probability is not None and probability >= settings["threshold"]:
                    outcome.record_generic(term["source"])

        sources = [term["source"] for term in kept]
        pairs, dropped = find_containment_pairs(sources, self.max_pairs)
        outcome.pairs_total = len(pairs)
        outcome.pairs_dropped = dropped
        if pairs and settings["relation_switch"]:
            def build_relation(index, pair):
                short_index, long_index = pair
                return (
                    {"a": kept[short_index], "b": kept[long_index]},
                    relation_question(index),
                )

            answers = self.engine.decide_batch(pairs, build_relation, purpose="term_relation")
            for index, (short_index, long_index) in enumerate(pairs):
                relation = read_relation(answers[index], index, settings["threshold"])
                if relation is None:
                    continue  # unknown keeps the pre-sweep behaviour for this pair
                key = (sources[short_index], sources[long_index])
                outcome.relations[key] = relation
                if relation == SUBORDINATE:
                    outcome.keep_separate.add(key)
                    outcome.subordinate.append(key)
                elif relation == UNRELATED:
                    outcome.keep_separate.add(key)
        return outcome

    def _build_generic(self, index: int, term: dict):
        return ({"source": term["source"], "kind": term["kind"]}, generic_question(index))

    # --- translation side: is the shared name rendered the same way? -----------

    def sweep_translations(self, entries, pairs) -> TranslationSweep:
        """entries: [{"source","translation"}]; pairs: [(shorter, longer)] to check.

        Only pairs that survived as two entries are worth checking, so the caller passes the
        subordinate pairs. A pair missing either translation is skipped: there is nothing to
        compare, and inventing a verdict would flag a term the sweep never saw.
        """
        outcome = TranslationSweep()
        if not self.settings["consistency_switch"]:
            return outcome
        by_source = {}
        for entry in entries or []:
            source = str((entry or {}).get("source", "")).strip()
            if source:
                by_source[source] = str((entry or {}).get("translation", "")).strip()

        checkable = []
        for short_source, long_source in pairs or []:
            short_translation = by_source.get(short_source, "")
            long_translation = by_source.get(long_source, "")
            if not short_translation or not long_translation:
                outcome.skipped += 1
                continue
            checkable.append({
                "a": {"source": short_source, "translation": short_translation},
                "b": {"source": long_source, "translation": long_translation},
            })

        if not checkable:
            return outcome
        answers = self.engine.decide_batch(checkable, self._build_consistency, purpose="term_consistency")
        for index, pair in enumerate(checkable):
            probability = read_consistency(answers[index], index)
            if probability is None:
                outcome.skipped += 1
                continue
            outcome.judged += 1
            if probability < self.settings["threshold"]:
                outcome.inconsistent.append((pair["a"]["source"], pair["b"]["source"], probability))
        return outcome

    def _build_consistency(self, index: int, pair: dict):
        return (pair, consistency_question(index))

# --- grouping -----------------------------------------------------------------

def group_sources(raw_grouped_inputs, keep_separate=None, enable_short_name_merge: bool = True):
    """Decide which extracted sources share one glossary entry.

    This is the rule the sweep exists to correct, so it lives here rather than inside the Qt
    task and can be tested directly. A shorter source joins a longer one only when it is a
    substring AND the sweep did not rule the pair a derived name. A pair with no verdict stays
    mergeable, which is exactly the behaviour that existed before the sweep.

    Returns (grouped_inputs, source_aliases), both keyed by the surviving primary source.
    """
    keep_separate = set(keep_separate or ())
    raw_grouped_inputs = raw_grouped_inputs or {}

    if not enable_short_name_merge:
        grouped_inputs = {
            source: {
                "source": source,
                "merged_sources": [source],
                "candidates": list(item.get("candidates", [])),
            }
            for source, item in raw_grouped_inputs.items()
        }
        return grouped_inputs, {source: source for source in grouped_inputs}

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
            if other_source not in source:
                continue
            if (other_source, source) in keep_separate:
                # A derived name, not a variant: it keeps its own entry, so standalone
                # occurrences of the short term are still covered by the glossary.
                continue
            merged_group["merged_sources"].append(other_source)
            merged_group["candidates"].extend(raw_grouped_inputs[other_source].get("candidates", []))
            source_aliases[other_source] = source
            consumed_sources.add(other_source)

    return grouped_inputs, source_aliases

