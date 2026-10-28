"""Clips must start and end on sentence boundaries even when units were split at pauses.

Live finding (CopyrightX lecture, 2026-10-07): unit u0031 "As you saw in the last lecture, and as you'll
see frequently in future lectures," + 1.5 s pause + u0032 "crucial doctrines in copyright law remain
ambiguous." — the model started a clip on u0032, i.e. mid-sentence.
"""

from __future__ import annotations

from clipper.ai.prompts import format_units
from clipper.ai.selection import Window, complete_sentences, validate_candidates
from clipper.ai.transcript import build_units, word_ids_for
from clipper.models import ProjectOptions, Segment, Transcript, Word


def _transcript(sentences: list[tuple[str, float]]) -> Transcript:
    """Words 0.3 s apart; each (text, gap_after_seconds) chunk is spoken in sequence."""
    words: list[Word] = []
    t = 0.0
    raw: list[tuple[str, float, float]] = []
    for text, gap in sentences:
        for token in text.split():
            raw.append((token, t, t + 0.25))
            t += 0.3
        t += gap
    ids = word_ids_for([r[1] for r in raw])
    for (token, a, b), wid in zip(raw, ids):
        words.append(Word(id=wid, text=token, raw=token.strip(".,"), start=a, end=b, segment_id="s00001",
                          chunk_index=0))
    seg = Segment(id="s00001", start=0.0, end=t, text=" ".join(w.text for w in words), word_start=0,
                  word_end=len(words) - 1, chunk_index=0)
    return Transcript(provider="test", model="test", language="en", duration=t + 1, words=words,
                      segments=[seg], units=build_units(words), chunks=[], created_at=0.0)


LECTURE = [
    ("The next role of theory is more instrumental.", 1.0),
    ("As you saw in the last lecture, and as you will see frequently in future lectures,", 1.5),  # pause split
    ("crucial doctrines in copyright law remain ambiguous.", 0.4),
    ("As a result lawyers arguing for their clients frequently make normative arguments.", 0.4),
    ("It is not enough to know what the law is.", 0.4),
    ("You have to be able to argue about what it should be.", 1.0),
]


def _idx(t: Transcript, startswith: str) -> int:
    return next(i for i, u in enumerate(t.units) if u.text.startswith(startswith))


def test_units_split_at_the_pause_mid_sentence():
    t = _transcript(LECTURE)
    i = _idx(t, "crucial")
    assert t.units[i - 1].text.endswith("lectures,")      # the precondition of the live bug


def test_start_on_a_continuation_is_extended_to_the_sentence_start():
    t = _transcript(LECTURE)
    s, e = _idx(t, "crucial"), _idx(t, "You have")
    assert complete_sentences(t, s, e, 5.0, 60.0) == (s - 1, e)


def test_when_extension_does_not_fit_the_partial_sentence_is_trimmed():
    t = _transcript(LECTURE)
    s, e = _idx(t, "crucial"), _idx(t, "You have")
    from clipper.ai.transcript import range_for_units
    a, _ = range_for_units(t, t.units[s].id, t.units[e].id)
    _, b = range_for_units(t, t.units[s].id, t.units[e].id)
    hi = (b - a) + 0.5                                     # extending by a whole unit would be too long
    assert complete_sentences(t, s, e, 3.0, hi) == (s + 1, e)


def test_end_on_an_unfinished_sentence_is_extended():
    t = _transcript(LECTURE)
    s, e = _idx(t, "The next role"), _idx(t, "As you saw")
    assert complete_sentences(t, s, e, 3.0, 60.0) == (s, e + 1)


def test_validate_candidates_starts_clips_on_a_sentence():
    t = _transcript(LECTURE)
    s, e = _idx(t, "crucial"), _idx(t, "You have")
    window = Window(index=0, unit_start=0, unit_end=len(t.units), context_before=0, context_after=len(t.units))
    raw = [{"start_unit_id": t.units[s].id, "end_unit_id": t.units[e].id, "hook": "h", "payoff": "p",
            "topic": "law", "reason": "r", "is_ad": False, "is_housekeeping": False, "needs_prior_context": False,
            "scores": {"hook": 4, "completeness": 4, "value": 4, "emotion": 3},
            "title": "Knowing the law is not enough"}]
    opts = ProjectOptions(clip_count=1, min_seconds=5, max_seconds=60)
    accepted, rejected = validate_candidates(raw, window, t, opts)
    assert rejected == []
    assert accepted[0].start_unit_id == t.units[s - 1].id and accepted[0].end_unit_id == t.units[e].id


def test_prompt_marks_continuation_units():
    t = _transcript(LECTURE)
    lines = format_units(t.units).splitlines()
    i = _idx(t, "crucial")
    assert "] … crucial doctrines" in lines[i]
    assert "] … " not in lines[i - 1] and "] … " not in lines[0]
