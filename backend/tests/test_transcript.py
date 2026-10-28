"""Tests for clipper.ai.transcript (pure merging/alignment/units/validation) + a real FFmpeg chunk test."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from clipper.ai.interfaces import RawChunkResult, RawSegment, RawWord
from clipper.ai.transcript import (
    MIN_WORDS,
    align_punctuation,
    build_units,
    insufficient_speech_reason,
    is_sentence_end,
    merge_chunks,
    range_for_units,
    speech_stats,
    unit_index,
    validate_transcript,
    words_between,
)
from clipper.config import Settings
from clipper.models import Transcript, TranscriptChunk, Word
from clipper.procs import run
from conftest import requires_ffmpeg

VOCAB = ["clear", "ideas", "make", "short", "clips", "worth", "watching", "every", "single", "time"]


@dataclass(frozen=True)
class Plan:
    """Duck-typed stand-in for clipper.media.audio.ChunkPlan (same fields)."""
    index: int
    start: float
    end: float
    keep_start: float
    keep_end: float


def meta_for(plans: list[Plan]) -> list[TranscriptChunk]:
    return [TranscriptChunk(index=p.index, start=p.start, end=p.end, keep_start=p.keep_start,
                            keep_end=p.keep_end, model="whisper-1") for p in plans]


def merge(results: list[tuple[Plan, RawChunkResult]], duration: float) -> Transcript:
    plans = [p for p, _ in results]
    t = merge_chunks(results, duration=duration, provider="openai", model="whisper-1", language="en",
                     chunk_meta=meta_for(plans))
    validate_transcript(t)
    return t


def ground_truth(blocks: int, *, per_block: int = 6, block: float = 4.0, period: float = 0.5,
                 on: float = 0.3) -> list[tuple[str, float, float, int]]:
    """(text, src_start, src_end, sentence_no): one sentence per block, then a pause."""
    out = []
    k = 0
    for b in range(blocks):
        for j in range(per_block):
            s = b * block + j * period
            out.append((VOCAB[k % len(VOCAB)], round(s, 6), round(s + on, 6), b))
            k += 1
    return out


def sentence_text(words: list[str]) -> str:
    return " " + " ".join(words).capitalize() + "."


def provider_view(gt: list[tuple[str, float, float, int]], plan: Plan, *, jitter: float = 0.0) -> RawChunkResult:
    """What a provider would return for the chunk audio [plan.start, plan.end] (chunk-relative times).

    Words fully inside the audio are returned; segments are returned for every sentence that has a
    word in the audio (their text always holds the whole sentence, like a real provider that heard a
    clipped sentence and still segmented it).
    """
    words: list[RawWord] = []
    by_sentence: dict[int, list[tuple[str, float, float, int]]] = {}
    for w in gt:
        if w[1] >= plan.start - 1e-9 and w[2] <= plan.end + 1e-9:
            words.append(RawWord(text=w[0], start=w[1] - plan.start + jitter, end=w[2] - plan.start + jitter))
            by_sentence.setdefault(w[3], []).append(w)
    segments = []
    for sno, ws in sorted(by_sentence.items()):
        full = [w for w in gt if w[3] == sno]
        segments.append(RawSegment(text=sentence_text([w[0] for w in full]),
                                   start=max(0.0, ws[0][1] - plan.start), end=ws[-1][2] - plan.start,
                                   no_speech_prob=0.01, avg_logprob=-0.2))
    return RawChunkResult(words=words, segments=segments, language="english",
                          duration=plan.end - plan.start, request_id=f"req_{plan.index}")


def plans_for(cuts: list[float], duration: float, overlap: float) -> list[Plan]:
    edges = [0.0, *cuts, duration + 1.0]
    plans = []
    for i in range(len(edges) - 1):
        ks, ke = edges[i], edges[i + 1]
        plans.append(Plan(index=i, start=max(0.0, ks - overlap), end=min(duration, ke + overlap),
                          keep_start=ks, keep_end=ke))
    return plans


def owner_index(plans: list[Plan], mid: float) -> int:
    return next(p.index for p in plans if p.keep_start <= mid < p.keep_end)


def mk_words(spec: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(id=f"w{i:06d}", text=t, raw=t.strip(".,?!"), start=s, end=e, segment_id="s00001", chunk_index=0)
            for i, (t, s, e) in enumerate(spec, start=1)]


def evenly(texts: list[str], *, start: float = 0.0, step: float = 0.4, dur: float = 0.3) -> list[Word]:
    return mk_words([(t, start + i * step, start + i * step + dur) for i, t in enumerate(texts)])


# ============================================================================ align_punctuation
ALIGN_CASES = [
    (["hello", "world"], " Hello, world.", ["Hello,", "world."]),
    (["hello"], ' "Hello," she said.', ['"Hello,"']),
    (["don’t", "stop"], " Don't stop!", ["Don't", "stop!"]),
    (["I", "dont", "know"], " I don't know.", ["I", "dont", "know."]),
    (["it", "cost", "1000"], " It cost 1,000.", ["It", "cost", "1000."]),
    (["it", "cost", "1,000"], " It cost 1000.", ["It", "cost", "1,000."]),
    (["a", "well", "known", "fact"], " A well-known fact.", ["A", "well", "known", "fact."]),
    (["well", "known"], " Well-known, really.", ["Well", "known,"]),
    (["well-known"], " Well known.", ["Well-known."]),
    (["so", "um", "yeah"], " So yeah.", ["So", "um", "yeah."]),
    (["so", "yeah"], " So, um, yeah.", ["So,", "yeah."]),
    (["colour", "is", "nice"], " Color is nice?", ["Colour", "is", "nice?"]),
    (["the", "the", "the"], " The, the, the.", ["The,", "the,", "the."]),
    (["hi"], "", ["hi"]),
    ([], " Hi.", []),
    (["U.S.", "policy"], " U.S. policy.", ["U.S.", "policy."]),
    (["she", "said", "wait"], " She said — wait...", ["She", "said—", "wait..."]),
    (["rock", "&", "roll"], " Rock & roll.", ["Rock", "&", "roll."]),
    ([" hi", "there "], " Hi there.", ["Hi", "there."]),
    (["gonna", "go"], " Going to go.", ["Gonna", "go."]),
    (["see", "ya"], " See you later.", ["See", "ya."]),
    (["he", "asked", "why"], ' He asked, "Why?"', ["He", "asked,", '"Why?"']),
    (["café", "olé"], " Café, olé!", ["Café,", "olé!"]),
    (["HELLO", "there"], " Hello there.", ["Hello", "there."]),
    (["it's", "50"], " It's 50%.", ["It's", "50%."]),
    (["well", "known", "really"], " Well-known, really.", ["Well", "known,", "really."]),
    (["it", "costs", "1", "000", "dollars"], " It costs 1,000 dollars.", ["It", "costs", "1", "000", "dollars."]),
    (["okay", "so", "well", "known", "stuff"], " OK, so well-known stuff!",
     ["Okay,", "so", "well", "known", "stuff!"]),
    (["one", "two", "three"], " Four five.", ["One", "two", "three."]),
]


@pytest.mark.parametrize("raw,segment,expected", ALIGN_CASES)
def test_align_punctuation_table(raw: list[str], segment: str, expected: list[str]) -> None:
    assert align_punctuation(raw, segment) == expected


def test_align_punctuation_always_returns_one_entry_per_word() -> None:
    for raw, segment, _ in ALIGN_CASES:
        out = align_punctuation(raw, segment)
        assert len(out) == len(raw)
        assert all(o.strip() for o, r in zip(out, raw) if r.strip())


# ============================================================================ units
def test_sentence_end_detection() -> None:
    for t in ["done.", "why?", "wow!", "wait…", "so...", '"stop."', "it.”", "(really?)", "I.", "I.”"]:
        assert is_sentence_end(t), t
    for t in ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "vs.", "e.g.", "i.e.", "etc.", "U.S.", "a.m.",
              "p.m.", "J.", "U.K.", "hello", "well,", "3.5", ""]:
        assert not is_sentence_end(t), t


def test_units_break_on_sentences_not_abbreviations() -> None:
    words = evenly("I met Mr. Smith and Dr. Jones at 9 a.m. in the U.S. today. J. R. Tolkien wrote "
                   "books, e.g. novels. Really? Yes!".split())
    units = build_units(words)
    assert [u.text for u in units] == [
        "I met Mr. Smith and Dr. Jones at 9 a.m. in the U.S. today.",
        "J. R. Tolkien wrote books, e.g. novels.",
        "Really?",
        "Yes!",
    ]
    assert [u.id for u in units] == ["u0001", "u0002", "u0003", "u0004"]
    for u in units:
        assert u.start == words[u.word_start].start and u.end == words[u.word_end].end


def test_unit_ends_after_the_pronoun_i() -> None:
    # "I." is the pronoun ending a sentence, not an initial: a clip must be able to end on "...so did I."
    display = align_punctuation(["so", "did", "i", "but", "then", "we", "left"], " So did I. But then we left.")
    assert display == ["So", "did", "I.", "But", "then", "we", "left."]
    units = build_units(evenly(display))
    assert [u.text for u in units] == ["So did I.", "But then we left."]


def test_units_break_on_long_pause_only() -> None:
    words = mk_words([("so", 0.0, 0.3), ("we", 0.4, 0.6), ("went", 1.7, 2.0),     # 1.1 s gap: no break
                      ("home", 3.3, 3.6), ("and", 3.7, 3.9)])                       # 1.3 s gap: break
    units = build_units(words, pause_break=1.2)
    assert [(u.word_start, u.word_end) for u in units] == [(0, 2), (3, 4)]


def test_long_run_on_breaks_at_last_comma_within_limit() -> None:
    texts = [f"w{i}" for i in range(100)]
    texts[30] = "w30,"
    texts[40] = "w40,"
    units = build_units(evenly(texts), max_words=45)
    spans = [(u.word_start, u.word_end) for u in units]
    assert spans == [(0, 40), (41, 85), (86, 99)]


def test_long_run_on_without_usable_comma_hard_breaks() -> None:
    texts = [f"w{i}" for i in range(100)]
    texts[2] = "w2,"          # too early to be a sensible break (fragment of 3 words)
    units = build_units(evenly(texts), max_words=45)
    assert [(u.word_start, u.word_end) for u in units] == [(0, 44), (45, 89), (90, 99)]
    assert all(u.word_end - u.word_start + 1 <= 45 for u in units)


def test_build_units_edge_cases() -> None:
    assert build_units([]) == []
    one = build_units(evenly(["alone"]))
    assert len(one) == 1 and one[0].text == "alone"
    assert len(build_units(evenly(["a", "b", "c"]), max_words=1)) == 3
    with pytest.raises(ValueError):
        build_units(evenly(["a"]), max_words=0)


# ============================================================================ merge_chunks
def test_merge_three_overlapping_chunks_no_duplicates_exact_offsets() -> None:
    gt = ground_truth(12)                                  # 48 s, 72 words, 12 sentences
    duration = 48.0
    plans = plans_for([15.4, 31.4], duration, overlap=2.75)   # cuts at pause midpoints (2.8 → 4.0)
    assert plans[1].start == pytest.approx(12.65)
    results = [(p, provider_view(gt, p)) for p in plans]
    # the overlaps really do contain the same words in both chunks
    overlap_words = sum(1 for p, r in results for w in r.words
                        if not p.keep_start <= p.start + (w.start + w.end) / 2 < p.keep_end)
    assert overlap_words >= 8
    t = merge(results, duration)
    assert len(t.words) == len(gt)
    for w, g in zip(t.words, gt):
        assert w.raw == g[0]
        assert abs(w.start - g[1]) < 1e-9 and abs(w.end - g[2]) < 1e-9
        assert w.chunk_index == owner_index(plans, (g[1] + g[2]) / 2)
    # word ids are derived from start times (ms) so re-merges never shift them
    assert [w.id for w in t.words[:3]] == [f"w{round(g[1] * 1000):09d}" for g in gt[:3]]
    assert len(t.segments) == 12 and [s.id for s in t.segments[:2]] == ["s00001", "s00002"]
    assert [u.text for u in t.units][:2] == ["Clear ideas make short clips worth.",
                                             "Watching every single time clear ideas."]
    assert len(t.units) == 12
    assert t.fixes == {}
    assert [c.index for c in t.chunks] == [0, 1, 2]


def test_merge_input_order_does_not_matter() -> None:
    gt = ground_truth(6)
    plans = plans_for([11.4], 24.0, overlap=2.0)
    results = [(p, provider_view(gt, p)) for p in plans]
    a = merge(results, 24.0)
    b = merge(list(reversed(results)), 24.0)
    assert [(w.text, w.start, w.end, w.chunk_index) for w in a.words] == \
           [(w.text, w.start, w.end, w.chunk_index) for w in b.words]


@pytest.mark.parametrize("cut,owners", [(10.0, [0, 1, 1]), (10.1, [0, 0, 1])])
def test_word_straddling_cut_is_owned_by_midpoint_once(cut: float, owners: list[int]) -> None:
    # "beta" [9.8, 10.3] straddles the cut; its midpoint 10.05 decides the owner. Both chunks hear it.
    src = [("alpha", 9.2, 9.7), ("beta", 9.8, 10.3), ("gamma", 10.4, 10.8)]
    plans = [Plan(0, 0.0, cut + 2.0, 0.0, cut), Plan(1, cut - 2.0, 20.0, cut, 21.0)]
    results = []
    for p in plans:
        words = [RawWord(t, s - p.start, e - p.start) for t, s, e in src]
        segs = [RawSegment(" Alpha beta gamma.", 9.2 - p.start, 10.8 - p.start)]
        results.append((p, RawChunkResult(words=words, segments=segs)))
    t = merge(results, 20.0)
    assert [w.text for w in t.words] == ["Alpha", "beta", "gamma."]
    assert [w.chunk_index for w in t.words] == owners
    assert [(w.start, w.end) for w in t.words] == [pytest.approx((s, e)) for _, s, e in src]
    # the segment straddling the cut is reported once, holding all three words
    assert len(t.segments) == 1
    assert (t.segments[0].word_start, t.segments[0].word_end) == (0, 2)
    assert t.segments[0].text == "Alpha beta gamma."


def test_midpoint_exactly_on_cut_belongs_to_later_chunk() -> None:
    plans = [Plan(0, 0.0, 6.0, 0.0, 5.0), Plan(1, 4.0, 10.0, 5.0, 11.0)]
    word = ("edge", 4.75, 5.25)               # midpoint 5.0 == cut
    results = [(p, RawChunkResult(words=[RawWord(word[0], word[1] - p.start, word[2] - p.start)],
                                  segments=[RawSegment(" Edge.", word[1] - p.start, word[2] - p.start)]))
               for p in plans]
    t = merge(results, 10.0)
    assert [(w.text, w.chunk_index) for w in t.words] == [("Edge.", 1)]


def test_seam_duplicates_with_provider_jitter_are_dropped() -> None:
    # The same spoken words near the cut get slightly different times from the two chunks, so
    # both copies pass the midpoint test; the seam de-duplication keeps only the first copy.
    plans = [Plan(0, 0.0, 12.0, 0.0, 10.0), Plan(1, 8.0, 20.0, 10.0, 21.0)]
    c0 = [("so", 9.40, 9.70), ("anyway", 9.72, 10.26)]          # "anyway" mid 9.99 → chunk 0
    c1 = [("anyway", 9.78, 10.30), ("next", 10.5, 10.9)]         # "anyway" mid 10.04 → chunk 1
    results = [
        (plans[0], RawChunkResult(words=[RawWord(t, s, e) for t, s, e in c0],
                                  segments=[RawSegment(" So anyway,", 9.4, 10.26)])),
        (plans[1], RawChunkResult(words=[RawWord(t, s - 8.0, e - 8.0) for t, s, e in c1],
                                  segments=[RawSegment(" anyway, next.", 1.78, 2.9)])),
    ]
    t = merge(results, 20.0)
    assert [w.raw for w in t.words] == ["so", "anyway", "next"]
    assert t.fixes.get("duplicate_words_dropped") == 1
    assert t.words[1].start == 9.72 and t.words[1].chunk_index == 0


def test_repeated_word_across_seam_is_not_mistaken_for_duplicate() -> None:
    plans = [Plan(0, 0.0, 12.0, 0.0, 10.0), Plan(1, 8.0, 20.0, 10.0, 21.0)]
    both = [("no", 9.6, 9.9), ("no", 10.1, 10.4)]
    results = [(p, RawChunkResult(words=[RawWord(t, s - p.start, e - p.start) for t, s, e in both],
                                  segments=[RawSegment(" No, no.", 9.6 - p.start, 10.4 - p.start)]))
               for p in plans]
    t = merge(results, 20.0)
    assert [(w.text, w.chunk_index) for w in t.words] == [("No,", 0), ("no.", 1)]
    assert "duplicate_words_dropped" not in t.fixes


def _seam_results(plans: list[Plan], views: list[list[tuple[str, float, float]]],
                  segments: list[tuple[str, float, float]]) -> list[tuple[Plan, RawChunkResult]]:
    """Each chunk's words (source times in ``views``) and segments, converted to chunk-relative times."""
    return [(p, RawChunkResult(words=[RawWord(t, s - p.start, e - p.start) for t, s, e in view],
                               segments=[RawSegment(txt, s - p.start, e - p.start) for txt, s, e in segments]))
            for p, view in zip(plans, views)]


def test_word_neither_chunk_owns_is_recovered_at_the_seam() -> None:
    # Mirror of the jitter case above: chunk 0 times "anyway" with its midpoint after the cut (10.04)
    # and chunk 1 with its midpoint before it (9.99), so each chunk leaves it to the other. The word
    # must not vanish: the earlier chunk's copy is kept (its own provider timing, never invented).
    plans = [Plan(0, 0.0, 12.0, 0.0, 10.0), Plan(1, 8.0, 20.0, 10.0, 21.0)]
    c0 = [("so", 9.40, 9.70), ("anyway", 9.78, 10.30), ("next", 10.5, 10.9)]
    c1 = [("so", 9.40, 9.70), ("anyway", 9.72, 10.26), ("next", 10.5, 10.9)]
    t = merge(_seam_results(plans, [c0, c1], [(" So anyway, next.", 9.4, 10.9)]), 20.0)
    assert [w.raw for w in t.words] == ["so", "anyway", "next"]
    assert t.fixes.get("seam_words_recovered") == 1
    assert "duplicate_words_dropped" not in t.fixes
    assert (t.words[1].start, t.words[1].end, t.words[1].chunk_index) == (9.78, 10.30, 0)
    assert [u.text for u in t.units] == ["So anyway, next."]


def test_word_lost_at_ideal_cut_without_silence_is_recovered() -> None:
    # A cut at exactly 600 s (no silence found); the two chunks time "hello" ~50 ms apart.
    plans = [Plan(0, 0.0, 603.0, 0.0, 600.0), Plan(1, 597.0, 900.0, 600.0, 901.0)]
    c0 = [("we", 599.4, 599.6), ("said", 599.62, 599.9), ("hello", 599.95, 600.10), ("there", 600.15, 600.4)]
    c1 = [("we", 599.4, 599.6), ("said", 599.62, 599.9), ("hello", 599.90, 600.05), ("there", 600.15, 600.4)]
    t = merge(_seam_results(plans, [c0, c1], [(" We said hello there.", 599.4, 600.4)]), 900.0)
    assert [w.raw for w in t.words] == ["we", "said", "hello", "there"]
    assert t.fixes.get("seam_words_recovered") == 1


@pytest.mark.parametrize("token,a,b", [
    ("anyway", (9.70, 10.00), (10.00, 10.30)),      # copies only touch (overlap 0, midpoints 0.3 s apart)
    ("the", (9.95, 9.95), (10.10, 10.10)),          # zero-length copies 0.15 s apart
])
def test_touching_or_zero_length_seam_copies_are_one_word(token: str, a: tuple[float, float],
                                                          b: tuple[float, float]) -> None:
    # Each chunk heard the word exactly once, so the two kept copies are one spoken word.
    plans = [Plan(0, 0.0, 12.0, 0.0, 10.0), Plan(1, 8.0, 20.0, 10.0, 21.0)]
    c0 = [("so", 9.40, 9.60), (token, *a), ("next", 10.5, 10.9)]
    c1 = [("so", 9.40, 9.60), (token, *b), ("next", 10.5, 10.9)]
    t = merge(_seam_results(plans, [c0, c1], [(f" So {token} next.", 9.4, 10.9)]), 20.0)
    assert [w.raw for w in t.words] == ["so", token, "next"]
    assert t.fixes.get("duplicate_words_dropped") == 1
    assert (t.words[1].start, t.words[1].end, t.words[1].chunk_index) == (*a, 0)


def test_quick_genuine_repeat_with_touching_times_is_kept() -> None:
    # "no no" spoken back to back across the cut: both chunks heard BOTH words, so the touching
    # copies at the seam are two words, not one word timed twice.
    plans = [Plan(0, 0.0, 12.0, 0.0, 9.9), Plan(1, 8.0, 20.0, 9.9, 21.0)]
    both = [("no", 9.6, 9.85), ("no", 9.85, 10.1)]
    t = merge(_seam_results(plans, [both, both], [(" No no.", 9.6, 10.1)]), 20.0)
    assert [(w.raw, w.chunk_index) for w in t.words] == [("no", 0), ("no", 1)]
    assert "duplicate_words_dropped" not in t.fixes and "seam_words_recovered" not in t.fixes


@pytest.mark.parametrize("seed", range(150))
def test_seam_in_continuous_speech_neither_loses_nor_duplicates_words(seed: int) -> None:
    # No silence at the cut (it lands mid-speech) and each chunk's timings are shifted by its own
    # offset, as two independent decodes are. Every spoken word must come out exactly once.
    rng = np.random.default_rng(seed)
    gt, t = [], 1.0
    while t < 20.0:
        dur = 0.0 if rng.random() < 0.15 else float(rng.uniform(0.05, 0.4))      # some zero-length words
        gt.append((f"tok{len(gt)}", t, t + dur))
        t += dur + float(rng.uniform(0.0, 0.05))
    cut = float(rng.uniform(8.0, 12.0))
    plans = [Plan(0, 0.0, cut + 3.0, 0.0, cut), Plan(1, cut - 3.0, 25.0, cut, 26.0)]
    results = []
    for p in plans:
        off = float(rng.normal(0.0, 0.06))
        words = [RawWord(w, s - p.start + off, e - p.start + off) for w, s, e in gt if s >= p.start and e <= p.end]
        results.append((p, RawChunkResult(words=words, segments=[RawSegment(" Speech.", 0.0, p.end - p.start)])))
    merged = merge(results, 25.0)
    assert sorted(w.raw for w in merged.words) == sorted(w for w, _, _ in gt)


def test_non_monotonic_provider_output_is_fixed_and_counted() -> None:
    plan = Plan(0, 0.0, 10.0, 0.0, 11.0)
    words = [RawWord("b", 1.0, 1.5), RawWord("a", 0.2, 0.6), RawWord("c", 1.4, 1.9),
             RawWord("d", 3.0, 2.5), RawWord("e", 1.6, 1.7)]
    t = merge([(plan, RawChunkResult(words=words, segments=[RawSegment(" A b c e d.", 0.2, 3.0)]))], 10.0)
    got = [(w.raw, w.start, w.end) for w in t.words]
    assert got == [("a", 0.2, 0.6), ("b", 1.0, 1.5), ("c", 1.5, 1.9), ("e", 1.9, 1.9), ("d", 3.0, 3.0)]
    assert t.fixes["inverted_times"] == 1
    assert t.fixes["overlapping_words"] == 2
    assert t.fixes["reordered_words"] > 0
    # never invent timings: every start/end is a provider value or a neighbour's boundary
    provider_times = {v for w in words for v in (w.start, w.end)}
    assert all(w.start in provider_times and w.end in provider_times for w in t.words)


def test_words_outside_duration_are_clamped_or_dropped() -> None:
    duration = 10.0
    plan = Plan(0, 0.0, duration, 0.0, duration + 1.0)
    words = [RawWord("early", -0.2, 0.3), RawWord("mid", 5.0, 5.4), RawWord("late", 9.9, 10.4),
             RawWord("beyond", 10.2, 10.5), RawWord("gone", 11.2, 11.6)]   # mid 11.4 ≥ keep_end → not ours
    t = merge([(plan, RawChunkResult(words=words, segments=[RawSegment(" Early mid late beyond gone.", 0, 11.6)]))],
              duration)
    assert [(w.raw, w.start, w.end) for w in t.words] == [
        ("early", 0.0, 0.3), ("mid", 5.0, 5.4), ("late", 9.9, 10.0), ("beyond", 10.0, 10.0)]
    assert t.fixes["clamped_times"] == 3
    assert all(0.0 <= s.start <= s.end <= duration for s in t.segments)


def test_empty_chunks_and_empty_words() -> None:
    plans = plans_for([10.0, 20.0], 30.0, overlap=1.0)
    words = [RawWord("hello", 2.0, 2.4), RawWord("  ", 2.5, 2.6), RawWord("there", 2.7, 3.0)]
    results = [(plans[0], RawChunkResult(words=words, segments=[RawSegment(" Hello there.", 2.0, 3.0)])),
               (plans[1], RawChunkResult(words=[], segments=[])),
               (plans[2], RawChunkResult(words=[RawWord("bye", 5.0, 5.3)], segments=[RawSegment(" Bye!", 5.0, 5.3)]))]
    t = merge(results, 30.0)
    assert [w.text for w in t.words] == ["Hello", "there.", "Bye!"]
    assert abs(t.words[2].start - (plans[2].start + 5.0)) < 1e-9
    assert t.fixes["empty_words_dropped"] == 1
    empty = merge([(p, RawChunkResult(words=[], segments=[])) for p in plans], 30.0)
    assert empty.words == [] and empty.segments == [] and empty.units == []


def test_chunk_without_segments_gets_a_synthetic_segment() -> None:
    plan = Plan(0, 0.0, 5.0, 0.0, 6.0)
    t = merge([(plan, RawChunkResult(words=[RawWord("no", 0.1, 0.3), RawWord("segments", 0.4, 0.9)],
                                     segments=[]))], 5.0)
    assert [w.text for w in t.words] == ["no", "segments"]
    assert len(t.segments) == 1 and t.segments[0].text == "no segments"
    assert t.fixes["synthetic_segments"] == 1


def test_segment_without_words_is_kept_once_by_midpoint() -> None:
    plans = [Plan(0, 0.0, 12.0, 0.0, 10.0), Plan(1, 8.0, 20.0, 10.0, 21.0)]
    results = []
    for p in plans:
        segs = [RawSegment(" Hi.", 1.0 - p.start, 1.3 - p.start), RawSegment(" (music)", 9.0 - p.start, 9.6 - p.start)]
        results.append((p, RawChunkResult(words=[RawWord("hi", 1.0 - p.start, 1.3 - p.start)]
                                           if p.start <= 1.0 else [], segments=segs)))
    t = merge(results, 20.0)
    assert [s.text for s in t.segments] == ["Hi.", "(music)"]
    assert (t.segments[1].word_start, t.segments[1].word_end) == (-1, -1)
    assert t.segments[1].chunk_index == 0


def test_invalid_numbers_are_dropped() -> None:
    plan = Plan(0, 0.0, 5.0, 0.0, 6.0)
    t = merge([(plan, RawChunkResult(words=[RawWord("ok", 0.1, 0.3), RawWord("bad", math.nan, 1.0)],
                                     segments=[RawSegment(" Ok.", 0.1, 0.3)]))], 5.0)
    assert [w.raw for w in t.words] == ["ok"] and t.fixes["invalid_words_dropped"] == 1


def test_merge_rejects_bad_duration() -> None:
    with pytest.raises(ValueError):
        merge_chunks([], duration=math.nan, provider="openai", model="whisper-1", language="en", chunk_meta=[])


def test_merge_accepts_real_chunk_plan() -> None:
    audio = pytest.importorskip("clipper.media.audio")
    plan = audio.ChunkPlan(index=0, start=0.0, end=4.0, keep_start=0.0, keep_end=5.0)
    t = merge_chunks([(plan, RawChunkResult(words=[RawWord("hi", 0.1, 0.4)], segments=[RawSegment(" Hi.", 0.1, 0.4)]))],
                     duration=4.0, provider="openai", model="whisper-1", language="en",
                     chunk_meta=meta_for([Plan(0, 0.0, 4.0, 0.0, 5.0)]))
    validate_transcript(t)
    assert t.words[0].text == "Hi."


# ============================================================================ validation
@pytest.fixture
def valid() -> Transcript:
    gt = ground_truth(4)
    plans = plans_for([7.4], 16.0, overlap=1.5)
    return merge([(p, provider_view(gt, p)) for p in plans], 16.0)


def _broken(t: Transcript, mutate) -> Transcript:
    c = t.model_copy(deep=True)
    mutate(c)
    return c


@pytest.mark.parametrize("name,mutate", [
    ("word id", lambda t: setattr(t.words[1], "id", "w000009")),
    ("overlap", lambda t: setattr(t.words[2], "start", t.words[1].end - 0.05)),
    ("beyond duration", lambda t: setattr(t.words[-1], "end", t.duration + 1)),
    ("negative start", lambda t: setattr(t.words[0], "start", -0.5)),
    ("inverted", lambda t: setattr(t.words[3], "end", t.words[3].start - 0.01)),
    ("empty text", lambda t: setattr(t.words[0], "text", "  ")),
    ("nan time", lambda t: setattr(t.words[0], "end", math.nan)),
    ("segment id", lambda t: setattr(t.segments[0], "id", "s00002")),
    ("segment range gap", lambda t: setattr(t.segments[1], "word_start", t.segments[1].word_start + 1)),
    ("segment half range", lambda t: setattr(t.segments[0], "word_end", -1)),
    ("word segment id", lambda t: setattr(t.words[0], "segment_id", "s00002")),
    ("segment order", lambda t: setattr(t.segments[1], "start", 0.0) or setattr(t.segments[0], "start", 1.0)),
    ("unit id", lambda t: setattr(t.units[0], "id", "u0000")),
    ("unit gap", lambda t: setattr(t.units[1], "word_start", t.units[1].word_start + 1)),
    ("unit times", lambda t: setattr(t.units[0], "end", t.units[0].end + 0.5)),
    ("unit missing", lambda t: t.units.pop()),
    ("duplicate chunk", lambda t: t.chunks.append(t.chunks[0])),
    ("bad duration", lambda t: setattr(t, "duration", -1.0)),
])
def test_validate_transcript_rejects(valid: Transcript, name: str, mutate) -> None:
    validate_transcript(valid)
    with pytest.raises(ValueError):
        validate_transcript(_broken(valid, mutate))


# ============================================================================ lookups
def test_words_between_uses_midpoints(valid: Transcript) -> None:
    w0, w1 = valid.words[0], valid.words[1]                    # [0, .3], [.5, .8]
    assert words_between(valid, 0.0, 0.15) == []               # w0 midpoint is 0.15 (half-open end)
    assert words_between(valid, 0.0, 0.16) == [w0]
    assert words_between(valid, 0.15, 0.65) == [w0]
    assert words_between(valid, 0.15, 0.66) == [w0, w1]
    assert words_between(valid, 5.0, 5.0) == []
    assert len(words_between(valid, 0.0, valid.duration)) == len(valid.words)


def test_range_for_units_padding_and_clamping() -> None:
    words = mk_words([("First.", 0.1, 0.5), ("Second", 0.7, 1.0), ("one.", 1.1, 1.4),
                      ("Third.", 3.4, 3.8), ("Last.", 3.9, 4.95)])
    t = Transcript(provider="openai", model="whisper-1", language="en", duration=5.0, words=words,
                   segments=[], units=build_units(words), chunks=[], created_at=0.0)
    assert [u.text for u in t.units] == ["First.", "Second one.", "Third.", "Last."]
    assert unit_index(t) == {"u0001": 0, "u0002": 1, "u0003": 2, "u0004": 3}
    assert range_for_units(t, "u0001", "u0001") == (0.0, pytest.approx(0.6))       # start clamped to 0
    s, e = range_for_units(t, "u0002", "u0002")
    assert s == pytest.approx(0.6) and e == pytest.approx(1.65)                      # half gap / full pad
    s, e = range_for_units(t, "u0002", "u0003")
    assert s == pytest.approx(0.6) and e == pytest.approx(3.85)
    assert range_for_units(t, "u0004", "u0004") == (pytest.approx(3.85), 5.0)        # end clamped to duration
    assert range_for_units(t, "u0002", "u0002", pad=0.0) == (0.7, 1.4)
    with pytest.raises(ValueError):
        range_for_units(t, "u0003", "u0002")
    with pytest.raises(ValueError):
        range_for_units(t, "u0001", "u0099")


def _clip_texts(t: Transcript, unit: str) -> tuple[list[str], list[str]]:
    """(words, caption cue texts) of a clip cut at ``range_for_units(unit, unit)``."""
    from clipper.models import ClipEdit
    from clipper.render.captions import build_cues
    s, e = range_for_units(t, unit, unit)
    return [w.text for w in words_between(t, s, e)], [c.text for c in build_cues(t.words, ClipEdit(src_start=s,
                                                                                                    src_end=e))]


def test_zero_length_last_word_stays_in_its_units_clip() -> None:
    # whisper-style zero-length "it." with the next word starting at the same instant: the padding is
    # 0, so a half-open range ending at last.end would drop "it." here and hand it to the next clip.
    words = mk_words([("We", 0.0, 0.3), ("can", 0.3, 0.6), ("do", 0.6, 0.9), ("it.", 0.9, 0.9),
                      ("So", 0.9, 1.3), ("go.", 1.3, 1.6)])
    t = _transcript_from(words, 2.0)
    assert [u.text for u in t.units] == ["We can do it.", "So go."]
    first_words, first_cues = _clip_texts(t, "u0001")
    assert first_words == ["We", "can", "do", "it."] and " ".join(first_cues) == "We can do it."
    second_words, second_cues = _clip_texts(t, "u0002")
    assert second_words == ["So", "go."] and " ".join(second_cues) == "So go."
    s, e = range_for_units(t, "u0001", "u0001")
    assert e - 0.9 <= 0.002 and range_for_units(t, "u0002", "u0002")[0] - 0.9 <= 0.002   # ~1 ms, not padding


def test_word_collapsed_to_zero_length_by_merge_stays_in_its_clip() -> None:
    # "it" overlaps the longer "do" and is collapsed to [1.0, 1.0] by the monotonic fix.
    plan = Plan(0, 0.0, 2.0, 0.0, 3.0)
    raw = [RawWord("we", 0.0, 0.3), RawWord("can", 0.3, 0.6), RawWord("do", 0.6, 1.0), RawWord("it", 0.8, 0.9),
           RawWord("so", 1.0, 1.3), RawWord("go", 1.3, 1.6)]
    t = merge([(plan, RawChunkResult(words=raw, segments=[RawSegment(" We can do it.", 0.0, 0.94),
                                                          RawSegment(" So go.", 0.94, 1.6)]))], 2.0)
    assert (t.words[3].text, t.words[3].start, t.words[3].end) == ("it.", 1.0, 1.0)
    assert _clip_texts(t, "u0001") == (["We", "can", "do", "it."], ["We can do it."])
    assert _clip_texts(t, "u0002")[0] == ["So", "go."]


# ============================================================================ speech amount
def _transcript_from(words: list[Word], duration: float) -> Transcript:
    return Transcript(provider="openai", model="whisper-1", language="en", duration=duration, words=words,
                      segments=[], units=build_units(words), chunks=[], created_at=0.0)


def test_speech_stats_bridges_short_pauses() -> None:
    words = mk_words([("a", 0.0, 0.3), ("b", 0.5, 0.8), ("c", 2.0, 2.5)])   # 0.2 s gap bridged, 1.2 s not
    stats = speech_stats(_transcript_from(words, 60.0))
    assert stats["word_count"] == 3
    assert stats["speech_seconds"] == pytest.approx(1.3)
    assert stats["coverage"] == pytest.approx(1.3 / 60, abs=1e-4)
    assert stats["words_per_minute"] == pytest.approx(3.0)


def test_enough_speech_returns_none() -> None:
    gt = ground_truth(14)                                  # 84 words in 56 s
    words = mk_words([(w[0], w[1], w[2]) for w in gt])
    assert insufficient_speech_reason(_transcript_from(words, 56.0), 56.0) is None


def test_too_few_words_is_explained() -> None:
    words = mk_words([(w[0], w[1], w[2]) for w in ground_truth(1)])
    reason = insufficient_speech_reason(_transcript_from(words, 48.0), 48.0)
    assert reason and reason.startswith("Only 6 words of speech were found")
    assert "speech-led" in reason
    one = insufficient_speech_reason(_transcript_from(words[:1], 48.0), 48.0)
    assert one and one.startswith("Only 1 word of speech was found")
    none = insufficient_speech_reason(_transcript_from([], 48.0), 48.0)
    assert none and none.startswith("No speech was found")


def test_low_coverage_is_explained() -> None:
    gt = ground_truth(20)                                  # 120 words in 80 s of a one-hour video
    words = mk_words([(w[0], w[1], w[2]) for w in gt])
    assert len(words) >= MIN_WORDS
    reason = insufficient_speech_reason(_transcript_from(words, 3600.0), 3600.0)
    assert reason and reason.startswith("Speech covers only about 2% of this video")
    assert "1 h 00 min" in reason and "speech-led" in reason


def test_slow_sparse_speech_is_explained() -> None:
    # 100 long "words" every 4 s: coverage is fine but only 15 words per minute.
    words = mk_words([(f"w{i}", i * 4.0, i * 4.0 + 2.0) for i in range(100)])
    reason = insufficient_speech_reason(_transcript_from(words, 400.0), 400.0)
    assert reason and "about 15 words per minute" in reason


def test_media_duration_falls_back_to_transcript_duration() -> None:
    words = mk_words([(w[0], w[1], w[2]) for w in ground_truth(14)])
    assert insufficient_speech_reason(_transcript_from(words, 56.0), 0.0) is None
    assert insufficient_speech_reason(_transcript_from(words, 56.0), math.nan) is None


# ============================================================================ real FFmpeg
SAMPLE_RATE = 16000


def _decode_pcm(path: Path, ffmpeg: str) -> np.ndarray:
    res = run([ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1",
               "-ar", str(SAMPLE_RATE), "-"], timeout=60)
    return np.frombuffer(res.stdout, dtype=np.int16).astype(np.float32)


def _detect_bursts(pcm: np.ndarray, *, hop: int = 40, threshold: float = 1000.0) -> list[tuple[float, float]]:
    """Tone bursts as (start, end) seconds relative to the start of ``pcm`` (2.5 ms resolution)."""
    n = len(pcm) // hop
    on = np.abs(pcm[: n * hop]).reshape(n, hop).max(axis=1) > threshold
    bursts, i = [], 0
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            bursts.append((i * hop / SAMPLE_RATE, (j + 1) * hop / SAMPLE_RATE))
            i = j + 1
        else:
            i += 1
    return bursts


def _fake_provider(bursts: list[tuple[float, float]], plan: Plan,
                   gt: list[tuple[str, float, float, int]]) -> RawChunkResult:
    """Label detected bursts like a provider would: words + one segment per pause-separated sentence."""
    words = []
    for s, e in bursts:
        nearest = min(gt, key=lambda g: abs((g[1] + g[2]) / 2 - (plan.start + (s + e) / 2)))
        words.append(RawWord(text=nearest[0], start=s, end=e))
    segments, group = [], []
    for w in words:
        if group and w.start - group[-1].end > 0.6:
            segments.append(RawSegment(sentence_text([g.text for g in group]), group[0].start, group[-1].end))
            group = []
        group.append(w)
    if group:
        segments.append(RawSegment(sentence_text([g.text for g in group]), group[0].start, group[-1].end))
    return RawChunkResult(words=words, segments=segments, language="english", duration=plan.end - plan.start)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_real_flac_chunks_merge_back_to_source_times(tmp_path: Path, settings: Settings) -> None:
    """Encode overlapping FLAC chunks with FFmpeg, 'transcribe' them by detecting the real tone bursts
    in each decoded chunk (chunk-relative times), merge, and compare with the ground truth."""
    duration = 32.0
    gt = ground_truth(8)                                   # bursts 0.3 s on, every 0.5 s; 1.2 s pauses
    wav = tmp_path / "speech.wav"
    expr = "0.5*sin(2*PI*440*t)*lt(mod(t,4),3)*lt(mod(mod(t,4),0.5),0.3)"
    ffmpeg = settings.ffmpeg
    run([ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"aevalsrc=exprs='{expr}':s={SAMPLE_RATE}:d={duration}", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
        timeout=60)
    # overlap 1.65 s puts chunk audio edges inside words → clipped partial words at every edge
    plans = plans_for([11.4, 23.4], duration, overlap=1.65)
    results = []
    for p in plans:
        flac = tmp_path / f"chunk_{p.index:03d}.flac"
        run([ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", str(wav),
             "-af", f"atrim=start={p.start}:end={p.end},asetpts=PTS-STARTPTS", "-c:a", "flac", str(flac)],
            timeout=60)
        assert flac.read_bytes()[:4] == b"fLaC"
        bursts = _detect_bursts(_decode_pcm(flac, ffmpeg))
        results.append((p, _fake_provider(bursts, p, gt)))
    # each chunk saw clipped words at its edges, and the overlaps hold words seen twice
    assert sum(len(r.words) for _, r in results) > len(gt) + 6
    t = merge(results, duration)
    assert len(t.words) == len(gt)
    for w, g in zip(t.words, gt):
        assert w.raw == g[0]
        assert abs(w.start - g[1]) < 0.006 and abs(w.end - g[2]) < 0.006, (w, g)
        assert w.chunk_index == owner_index(plans, (g[1] + g[2]) / 2)
    assert len(t.units) == 8 and all(u.text.endswith(".") for u in t.units)
    assert "duplicate_words_dropped" not in t.fixes
    assert insufficient_speech_reason(t, duration) is not None      # only 48 words: below the threshold


# ============================================================================ robustness
def _messy_chunk(rng: "np.random.Generator", plan: Plan) -> RawChunkResult:
    """Provider output with jitter, overlaps, disorder, odd segments and out-of-range values."""
    length = plan.end - plan.start
    words: list[RawWord] = []
    t = float(rng.uniform(-0.3, 0.5))
    while t < length + 0.5:
        dur = float(rng.uniform(0.0, 0.6))
        start = t + float(rng.normal(0, 0.08))
        end = start + dur if rng.random() > 0.05 else start - 0.1          # some inverted words
        text = str(rng.choice(VOCAB + ["", " ", "1,000", "don't", "well-known", "&"]))
        words.append(RawWord(text=text, start=start, end=end))
        t += float(rng.uniform(0.05, 0.9)) if rng.random() > 0.1 else float(rng.uniform(1.0, 2.5))
    if words and rng.random() < 0.3:
        words[int(rng.integers(len(words)))].start = math.nan
    order = list(range(len(words)))
    for _ in range(int(rng.integers(0, 4)) if len(order) > 1 else 0):     # a little disorder
        i, j = (int(x) for x in rng.integers(len(order), size=2))
        order[i], order[j] = order[j], order[i]
    words = [words[i] for i in order]
    segments: list[RawSegment] = []
    s = 0.0
    while s < length and rng.random() > 0.02:
        e = s + float(rng.uniform(0.5, 8.0))
        text = " " + " ".join(str(x) for x in rng.choice(VOCAB, size=int(rng.integers(0, 12)))) + \
               str(rng.choice([".", "?", ",", "", "..."]))
        segments.append(RawSegment(text=text, start=s + float(rng.normal(0, 0.2)), end=e))
        s = e + float(rng.uniform(-0.5, 1.0))
    if rng.random() < 0.15:
        segments = []
    return RawChunkResult(words=words, segments=segments)


@pytest.mark.parametrize("seed", range(60))
def test_merge_output_always_validates(seed: int) -> None:
    rng = np.random.default_rng(seed)
    duration = float(rng.uniform(5.0, 90.0))
    n_cuts = int(rng.integers(0, 4))
    cuts = sorted(float(c) for c in rng.uniform(1.0, duration - 1.0, size=n_cuts))
    plans = plans_for(cuts, duration, overlap=float(rng.uniform(0.0, 3.0)))
    results = [(p, _messy_chunk(rng, p)) for p in plans]
    t = merge(results, duration)                       # merge() runs validate_transcript
    assert all(w.text.strip() for w in t.words)
    assert len(t.words) <= sum(len(r.words) for _, r in results)
    assert all(0.0 <= w.start <= w.end <= duration for w in t.words)
    assert sum(u.word_end - u.word_start + 1 for u in t.units) == len(t.words)



def test_word_ids_are_stable_when_another_word_is_dropped() -> None:
    """Dropping one word on a re-merge must not change the ids of the other words, because the
    owner's caption corrections (word_overrides / cue_overrides) are keyed by word id."""
    from clipper.ai.transcript import word_ids_for

    starts = [0.0, 0.5, 1.0, 1.5]
    full = word_ids_for(starts)
    without_second = word_ids_for([0.0, 1.0, 1.5])
    assert full == ["w000000000", "w000000500", "w000001000", "w000001500"]
    assert without_second == [full[0], full[2], full[3]]
    same_ms = word_ids_for([2.0, 2.0, 2.0004, 3.0])
    assert same_ms == ["w000002000", "w000002000-2", "w000002000-3", "w000003000"]
    assert same_ms == sorted(same_ms)            # lexical order follows time order
    assert all(len(i) <= 16 for i in same_ms)    # API limit for word ids (CaptionEditIn)
