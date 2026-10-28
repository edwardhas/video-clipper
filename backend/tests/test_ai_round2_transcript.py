"""Round-2 regression tests for clipper.ai.transcript.merge_chunks.

* Silence hallucinations: a word wholly inside a detected pause whose provider segment says
  ``no_speech_prob >= 0.5`` is dropped (both conditions), with a real FFmpeg ``silencedetect`` case.
* Seam recovery is not blocked by another occurrence of a common word nearby.
* Two timings of one seam word are merged even when the same word was said shortly before
  (copies are paired by their order among the copies both chunks heard).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from clipper.ai.interfaces import RawChunkResult, RawSegment, RawWord
from clipper.ai.transcript import merge_chunks, validate_transcript
from clipper.config import Settings
from clipper.media.audio import detect_silences
from clipper.models import TRANSCRIPT_VERSION, Transcript
from clipper.procs import run
from conftest import requires_ffmpeg
from test_transcript import Plan, meta_for

SPEECH = 0.02          # typical no_speech_prob of a real speech segment


def merge(results: list[tuple[Plan, RawChunkResult]], duration: float,
          silences: list[tuple[float, float]] | None) -> Transcript:
    t = merge_chunks(results, duration=duration, provider="openai", model="whisper-1", language="en",
                     chunk_meta=meta_for([p for p, _ in results]), silences=silences)
    validate_transcript(t)
    return t


def chunk(plan: Plan, segments: list[tuple[str, float, float, float | None, list[tuple[str, float, float]]]]
          ) -> tuple[Plan, RawChunkResult]:
    """``segments``: (text, start, end, no_speech_prob, words) in source seconds → chunk-relative result."""
    words, segs = [], []
    for text, s, e, nsp, ws in segments:
        segs.append(RawSegment(text, s - plan.start, e - plan.start, no_speech_prob=nsp,
                               avg_logprob=-0.51 if nsp is not None and nsp >= 0.5 else -0.2))
        words.extend(RawWord(w, ws_ - plan.start, we - plan.start) for w, ws_, we in ws)
    return plan, RawChunkResult(words=words, segments=segs, language="english", duration=plan.end - plan.start)


ONE = Plan(0, 0.0, 30.0, 0.0, 31.0)
BEFORE = (" We talked about it.", 1.0, 2.6, SPEECH,
          [("We", 1.0, 1.2), ("talked", 1.25, 1.6), ("about", 1.65, 2.0), ("it.", 2.05, 2.6)])
AFTER = (" So anyway, next.", 18.0, 19.5, SPEECH, [("So", 18.0, 18.2), ("anyway,", 18.25, 18.8), ("next.", 19.0, 19.5)])


# ============================================================================ silence hallucinations
def test_lone_you_hallucinated_in_a_long_silence_is_dropped() -> None:
    # The real case: whisper returned a lone "you" (no_speech_prob 0.94, avg_logprob -0.51) in the
    # middle of a 15 s pause that silencedetect found at -35 dB / 0.35 s.
    you = (" you", 10.0, 10.4, 0.94, [("you", 10.0, 10.4)])
    results = [chunk(ONE, [BEFORE, you, AFTER])]
    t = merge(results, 25.0, silences=[(2.8, 17.8)])
    assert [w.raw for w in t.words] == ["We", "talked", "about", "it.", "So", "anyway,", "next."]
    assert t.fixes.get("dropped_in_silence") == 1
    assert [s.text for s in t.segments] == ["We talked about it.", "So anyway, next."]   # its segment went too
    # never moved: the remaining words keep the provider's timings exactly
    assert [(w.start, w.end) for w in t.words] == [(s, e) for _, s, e in BEFORE[4] + AFTER[4]]
    assert [u.text for u in t.units] == ["We talked about it.", "So anyway, next."]
    assert t.version == TRANSCRIPT_VERSION == 2
    # Without detected silences nothing changes (the parameter is optional).
    plain = merge(results, 25.0, silences=None)
    assert "you" in [w.raw for w in plain.words] and "dropped_in_silence" not in plain.fixes


def test_quiet_real_word_inside_a_short_silence_is_kept() -> None:
    # A soft "yes" below -35 dB lies wholly inside a short detected pause, but the provider was
    # confident it is speech (low no_speech_prob): it must stay.
    yes = (" Yes.", 12.2, 12.5, 0.12, [("Yes.", 12.2, 12.5)])
    t = merge([chunk(ONE, [BEFORE, yes, AFTER])], 25.0, silences=[(2.8, 11.9), (12.0, 12.9), (13.0, 17.8)])
    assert "Yes." in [w.text for w in t.words]
    assert "dropped_in_silence" not in t.fixes


@pytest.mark.parametrize("word", [(20.0, 20.3), (19.85, 20.2), (17.7, 17.9)])
def test_high_no_speech_word_outside_or_across_a_silence_is_kept(word: tuple[float, float]) -> None:
    # no_speech_prob is high but the word is not wholly inside a pause (outside, or reaching more
    # than 0.05 s past its edge): there is audible sound under it, so it is kept.
    okay = (" Okay.", word[0], word[1], 0.8, [("Okay.", *word)])
    t = merge([chunk(ONE, [BEFORE, okay])], 25.0, silences=[(2.8, 17.8), (19.6, 19.9)])
    assert "Okay." in [w.text for w in t.words]
    assert "dropped_in_silence" not in t.fixes


@pytest.mark.parametrize("word,dropped", [((5.97, 6.3), True), ((5.9, 6.3), False),
                                          ((13.7, 14.04), True), ((13.7, 14.1), False), ((9.0, 9.0), True)])
def test_silence_containment_has_a_small_tolerance(word: tuple[float, float], dropped: bool) -> None:
    you = (" you", word[0], word[1], 0.9, [("you", *word)])
    t = merge([chunk(ONE, [BEFORE, you, AFTER])], 25.0, silences=[(6.0, 14.0)])
    assert ("you" not in [w.raw for w in t.words]) is dropped
    assert t.fixes.get("dropped_in_silence", 0) == int(dropped)


def test_unknown_no_speech_prob_is_treated_as_speech() -> None:
    you = (" you", 10.0, 10.4, None, [("you", 10.0, 10.4)])
    t = merge([chunk(ONE, [BEFORE, you, AFTER])], 25.0, silences=[(2.8, 17.8)])
    assert "you" in [w.raw for w in t.words]


def test_segment_keeps_only_its_real_words_and_a_wordless_hallucination_is_not_published() -> None:
    # One segment ("Okay, you") mixes a real word before the pause and a hallucinated one inside it;
    # another segment ("Thank you.") came back with no word timings at all, inside the pause.
    mixed = (" Okay, you", 3.0, 11.3, 0.7, [("Okay,", 3.0, 3.4), ("you", 11.0, 11.3)])
    wordless = (" Thank you.", 13.0, 14.0, 0.9, [])
    t = merge([chunk(ONE, [BEFORE, mixed, wordless, AFTER])], 25.0, silences=[(4.0, 17.8)])
    assert [w.raw for w in t.words] == ["We", "talked", "about", "it.", "Okay,", "So", "anyway,", "next."]
    assert [s.text for s in t.segments] == ["We talked about it.", "Okay,", "So anyway, next."]
    assert t.fixes.get("dropped_in_silence") == 1 and t.fixes.get("segments_dropped_in_silence") == 1


def test_hallucination_at_a_seam_in_a_silence_is_not_recovered() -> None:
    # Chunks are cut at the middle of a pause, where whisper likes to hallucinate. Each chunk timed its
    # "you" into the other's keep range; seam recovery must not bring the hallucination back.
    plans = [Plan(0, 0.0, 12.0, 0.0, 10.0), Plan(1, 8.0, 20.0, 10.0, 21.0)]
    real0 = (" Right.", 8.2, 8.6, SPEECH, [("Right.", 8.2, 8.6)])
    real1 = (" Next.", 11.6, 12.0, SPEECH, [("Next.", 11.6, 12.0)])
    results = [chunk(plans[0], [real0, (" you", 9.9, 10.2, 0.9, [("you", 9.9, 10.2)])]),
               chunk(plans[1], [(" you", 9.8, 10.1, 0.9, [("you", 9.8, 10.1)]), real1])]
    before = merge(results, 20.0, silences=None)
    assert [w.raw for w in before.words] == ["Right.", "you", "Next."]       # recovered without silences
    t = merge(results, 20.0, silences=[(8.8, 11.3)])
    assert [w.raw for w in t.words] == ["Right.", "Next."]
    assert "seam_words_recovered" not in t.fixes


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_real_silencedetect_pauses_drop_only_the_hallucination(tmp_path: Path, settings: Settings) -> None:
    """Loud speech-like tone, a 12 s pause holding a -50 dBFS murmur, then tone again. FFmpeg's
    silencedetect (-35 dB / 0.35 s, as the audio stage runs it) finds the pause; the provider's
    hallucinated "you" in it is dropped and the quiet word it was sure about is kept."""
    wav = tmp_path / "speech.wav"
    loud = "0.5*sin(2*PI*440*t)*(lt(t,3)+gte(t,15))"
    murmur = "0.003*sin(2*PI*300*t)*gte(t,7)*lt(t,7.5)"        # about -50 dBFS: below the -35 dB threshold
    run([settings.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"aevalsrc=exprs='{loud}+{murmur}':s=16000:d=18", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
        timeout=60)
    silences = detect_silences(wav, settings=settings)
    assert len(silences) == 1 and abs(silences[0][0] - 3.0) < 0.05 and abs(silences[0][1] - 15.0) < 0.05
    plan = Plan(0, 0.0, 18.0, 0.0, 19.0)
    results = [chunk(plan, [
        (" We talked.", 0.5, 2.9, SPEECH, [("We", 0.5, 1.0), ("talked.", 1.2, 2.9)]),
        (" Yes.", 7.0, 7.5, 0.1, [("Yes.", 7.0, 7.5)]),                  # the murmur: quiet real speech
        (" you", 10.0, 10.3, 0.94, [("you", 10.0, 10.3)]),               # hallucinated in digital silence
        (" Next.", 15.1, 16.0, SPEECH, [("Next.", 15.1, 16.0)]),
    ])]
    t = merge(results, 18.0, silences=silences)
    assert [w.raw for w in t.words] == ["We", "talked.", "Yes.", "Next."]
    assert t.fixes.get("dropped_in_silence") == 1


# ============================================================================ seam words
def _seam(views: list[list[tuple[str, float, float]]], text: str) -> Transcript:
    plans = [Plan(0, 0.0, 12.0, 0.0, 10.0), Plan(1, 8.0, 20.0, 10.0, 21.0)]
    results = [(p, RawChunkResult(words=[RawWord(w, s - p.start, e - p.start) for w, s, e in view],
                                  segments=[RawSegment(text, 9.0 - p.start, 11.0 - p.start)]))
               for p, view in zip(plans, views)]
    return merge(results, 20.0, silences=None)


def test_seam_word_is_recovered_when_the_same_common_word_was_kept_just_before() -> None:
    # "the cat and the dog": the 2nd "the" straddles the cut. Chunk 0 times every word +0.03 s (so
    # that "the" lands after the cut) and chunk 1 -0.05 s (before it): neither owns it. Chunk 0 kept
    # the 1st "the" 0.55 s earlier, a different occurrence, which must not block the recovery.
    truth = [("the", 9.40, 9.55), ("cat", 9.58, 9.85), ("and", 9.88, 9.97), ("the", 9.98, 10.06),
             ("dog", 10.10, 10.40)]
    views = [[(w, s + off, e + off) for w, s, e in truth] for off in (0.03, -0.05)]
    t = _seam(views, " The cat and the dog.")
    assert [w.text for w in t.words] == ["The", "cat", "and", "the", "dog."]
    assert t.fixes.get("seam_words_recovered") == 1 and "duplicate_words_dropped" not in t.fixes
    assert (t.words[3].start, t.words[3].chunk_index) == (10.01, 0)           # chunk 0's own timing


@pytest.mark.parametrize("words,seam_word", [
    (["the", "end", "of"], "the"),          # "the end of the day"
    (["I", "think"], "I"),                  # "I think I said"
])
def test_seam_copies_are_one_word_when_the_word_was_also_said_just_before(words: list[str],
                                                                          seam_word: str) -> None:
    # Zero-length copies of the seam word timed 9.95 (chunk 0) and 10.10 (chunk 1); the same word was
    # spoken < 0.5 s earlier and both chunks heard that too. The copies are the same (2nd) occurrence
    # in both chunks, so they are one word.
    lead = [(w, 9.40 + 0.12 * i, 9.40 + 0.12 * i + 0.1) for i, w in enumerate(words)]
    tail = [("day", 10.30, 10.60)]
    t = _seam([lead + [(seam_word, 9.95, 9.95)] + tail, lead + [(seam_word, 10.10, 10.10)] + tail], " Speech.")
    assert [w.raw for w in t.words] == words + [seam_word, "day"]
    assert t.fixes.get("duplicate_words_dropped") == 1
    assert (t.words[len(words)].start, t.words[len(words)].chunk_index) == (9.95, 0)


def test_genuine_repeat_across_the_seam_is_still_two_words() -> None:
    # "no, no" with zero-length timings: chunk 0 kept the 1st "no", chunk 1 kept the 2nd one.
    view = [("so", 9.30, 9.50), ("no", 9.80, 9.80), ("no", 10.15, 10.15), ("day", 10.40, 10.60)]
    t = _seam([view, view], " So no, no day.")
    assert [(w.raw, w.chunk_index) for w in t.words] == [("so", 0), ("no", 0), ("no", 1), ("day", 1)]
    assert "duplicate_words_dropped" not in t.fixes


def test_seams_in_continuous_speech_with_common_repeated_words() -> None:
    # The seam property of test_transcript.py with a 20-word Zipf vocabulary instead of unique tokens,
    # so the same word often recurs right at the cut. Before the round-2 fixes 33 of these 300 seams
    # lost or duplicated a word (now 2: cuts where neither occurrence order nor timing can tell the
    # copies apart).
    weights = 1.0 / np.arange(1, 21)
    weights /= weights.sum()
    bad = 0
    for seed in range(300):
        rng = np.random.default_rng(seed)
        gt, t = [], 1.0
        while t < 20.0:
            dur = 0.0 if rng.random() < 0.15 else float(rng.uniform(0.05, 0.4))
            gt.append((f"w{int(rng.choice(20, p=weights))}", t, t + dur))
            t += dur + float(rng.uniform(0.0, 0.05))
        cut = float(rng.uniform(8.0, 12.0))
        plans = [Plan(0, 0.0, cut + 3.0, 0.0, cut), Plan(1, cut - 3.0, 25.0, cut, 26.0)]
        results = []
        for p in plans:
            off = float(rng.normal(0.0, 0.06))
            words = [RawWord(w, s - p.start + off, e - p.start + off) for w, s, e in gt
                     if s >= p.start and e <= p.end]
            results.append((p, RawChunkResult(words=words, segments=[RawSegment(" Speech.", 0.0, p.end - p.start)])))
        merged = merge(results, 25.0, silences=None)
        bad += Counter(w.raw for w in merged.words) != Counter(w for w, _, _ in gt)
    assert bad <= 4
