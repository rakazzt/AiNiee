"""Ready-made decisions, expressed as typed questions.

Each helper turns one judgment AiNiee actually needs into ATOMIC noul questions, plus a
reader that maps the returned probabilities back into the shape the caller already uses.
They live here, with no Qt and no app config, so the decision logic can be tested on its
own and the Qt-dependent checker only has to wire it up.

Instructions are written in English on purpose: the primitives and their criteria are
documented that way, and the target/source language names are passed in as state values
rather than baked into the sentence.
"""

from ModuleFolders.Infrastructure.DecisionEngine import Questions

# A noul answer is a probability, so it has to be cut somewhere. 0.5 is "more likely yes
# than no"; the caller keeps its own, separate threshold for how many bad lines make a
# chunk bad, exactly as it did with the local detector.
TARGET_THRESHOLD = 0.5

# Reported when the text is neither the target language nor leftover source text.
UNKNOWN_LANGUAGE = "unknown"


def language_question_ids(index: int) -> tuple:
    """The (language, residual) question ids for one item index."""
    return "i{0}_lang".format(index), "i{0}_residual".format(index)


def language_questions(index: int) -> dict:
    """Two questions about one line: is it the target language, and is it still source?

    Asked separately rather than as one compound question, because a compound one hides
    which of the two failed - and these two fail for different reasons. A line can be in
    the wrong language without being untranslated, and it can be untranslated while
    looking like the target language (Chinese target, Japanese source and vice versa).
    """
    lang_id, residual_id = language_question_ids(index)
    return {
        lang_id: Questions.noul(
            "Is `items.{0}.translated` written in `target_language`?".format(index),
            true_meaning="The text is written in target_language.",
            false_meaning=(
                "The text is in some other language, or it is still the untranslated "
                "source text."
            ),
        ),
        residual_id: Questions.noul(
            "Does `items.{0}.translated` still contain a passage copied unchanged from "
            "`items.{0}.source`?".format(index),
            true_meaning="Part of the source text is still present, untranslated.",
            false_meaning="The source text has been translated throughout.",
        ),
    }


def language_verdict(answers, index: int, target_code: str, source_code: str):
    """Map one line's answers onto (language codes, confidence), or None if unreadable.

    The returned pair is deliberately the same shape the local detector produces, so the
    chunk statistics, the cache flags and the report below it need no changes.

    None means "could not tell", never "no": a caller that treated it as a failure would
    flag every line whenever the model was unavailable.
    """
    lang_id, residual_id = language_question_ids(index)
    in_target = Questions.noul_probability(answers, lang_id)
    if in_target is None:
        return None
    if in_target >= TARGET_THRESHOLD:
        return [target_code], in_target
    still_source = Questions.noul_probability(answers, residual_id)
    if still_source is not None and still_source >= TARGET_THRESHOLD:
        # Naming the source language is what makes the existing error row readable:
        # "language mismatch (detected: ja, target: zh)".
        return [source_code or UNKNOWN_LANGUAGE], still_source
    return [UNKNOWN_LANGUAGE], 1.0 - in_target
