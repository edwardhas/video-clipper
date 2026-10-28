"""Clip selection: transcript windows, structured candidate mining, validation, dedupe and review.

Flow (driven by ``clipper.jobs.pipeline``)::

    find_candidates  -> per window: Responses API + Structured Outputs -> validate_candidates
                     -> dedupe_candidates -> renumber (c001, c002, ...)
    shortlist        -> top candidates by model_score (topic-diverse tie-break)
    review           -> batches of <= 8 candidates with <= 2 low-detail frames each
                        (each paid window / batch result — and each paid attempt inside one —
                        goes into the optional ``JsonCache`` before a cancel is honoured, so a
                        resume or retry never pays twice)
    apply_review     -> validate decisions, boundary nudges, duplicates, global dedupe,
                        threshold, rank, <= clip_count

The model only ever names unit IDs; every time below (``src_start``/``src_end``) is
derived in code from the transcript with ``range_for_units``. Fewer clips than
requested are returned when the source does not justify the count — never padded.
"""

from __future__ import annotations

import base64
import bisect
import hashlib
import logging
import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

import pydantic
from pydantic import BaseModel, ConfigDict, Field

from ..config import Settings
from ..errors import CancelledError, ClipperError, InterruptedError_, ProviderError
from ..models import (
    MAX_CLIP_SECONDS, MIN_CLIP_SECONDS, Candidate, CandidateScores, ClipSelection, ProjectOptions, ReviewDecision,
    Transcript, Unit,
)
from ..procs import CancelToken
from ..storage import stable_hash
from . import prompts
from .interfaces import JsonCache, UsageSink
from .openai_client import call_openai, make_client
from .prompts import PROMPT_VERSION, REVIEW_PROMPT_VERSION
from .transcript import is_sentence_end, range_for_units, unit_index

log = logging.getLogger(__name__)

PROVIDER = "openai"
DURATION_LOW = 0.85            # accepted clip length = [min_seconds*0.85, max_seconds*1.15] ...
DURATION_HIGH = 1.15
MAX_ATTEMPTS = 3               # 1 request + 2 corrective retries
REVIEW_BATCH_SIZE = 8
FRAMES_PER_CANDIDATE = 2
BOUNDARY_SHIFT_UNITS = 3       # review may move start/end by up to this many units
REVIEW_CONTEXT_SECONDS = 20.0
TITLE_MAX_CHARS = 60
REVIEW_REASON_MAX_CHARS = 140
MAX_FRAME_BYTES = 4 * 1024 * 1024
RANGE_PAD = 0.25               # range_for_units' default edge padding (seconds)

# Weights for the code-derived model_score (sub-scores are 1..5).
SCORE_WEIGHTS = {"completeness": 0.35, "hook": 0.30, "value": 0.25, "emotion": 0.10}

# Rejection codes that mean "the model named unit IDs it was not allowed to use".
ID_REJECTIONS = frozenset({"unknown_unit_id", "outside_window", "reversed_range"})

T = TypeVar("T", bound=BaseModel)


# ============================================================================ structured output models
class _Strict(BaseModel):
    """Strict-Structured-Outputs compatible: every field required, no extra keys, no dicts."""
    model_config = ConfigDict(extra="forbid")


class CandidateScoresOut(_Strict):
    hook: int = Field(ge=1, le=5, description="1-5: strength of the first ~3 seconds")
    completeness: int = Field(ge=1, le=5, description="1-5: one self-contained idea with a payoff")
    value: int = Field(ge=1, le=5, description="1-5: insight, surprise or usefulness")
    emotion: int = Field(ge=1, le=5, description="1-5: humour, tension, passion or emotion")


class CandidateOut(_Strict):
    start_unit_id: str = Field(description="First unit of the clip, copied from transcript_selectable")
    end_unit_id: str = Field(description="Last unit of the clip (inclusive), copied from transcript_selectable")
    hook: str = Field(description="Opening line and why it stops the scroll (max 150 chars)")
    payoff: str = Field(description="Punchline, insight or conclusion at the end (max 150 chars)")
    topic: str = Field(description="2-4 word topic label")
    reason: str = Field(description="Why it works as a standalone short (max 200 chars)")
    is_ad: bool
    is_housekeeping: bool
    needs_prior_context: bool
    scores: CandidateScoresOut
    title: str = Field(description="Accurate title, max 60 chars, no clickbait")


class CandidateListOut(_Strict):
    candidates: list[CandidateOut]


class ReviewDecisionOut(_Strict):
    candidate_id: str
    reason: str = Field(description="Why it works or not (max 140 chars)")
    opening: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    interest: int = Field(ge=1, le=5)
    visual: int = Field(ge=1, le=5, description="Suitability of the frames for a vertical crop")
    final_score: int = Field(ge=0, le=100)
    keep: bool
    duplicate_of: str | None = Field(description="Id of a better candidate covering the same moment, or null")
    title: str = Field(description="Accurate title, max 60 chars")
    start_unit_id: str = Field(description="Original start id or one from the allowed start list")
    end_unit_id: str = Field(description="Original end id or one from the allowed end list")


class ReviewListOut(_Strict):
    decisions: list[ReviewDecisionOut]


# ============================================================================ windows
@dataclass(frozen=True)
class Window:
    """A slice of ``Transcript.units`` (all fields are unit indices, half-open ranges).

    selectable units  = units[unit_start:unit_end]
    context before    = units[context_before:unit_start]
    context after     = units[unit_end:context_after]
    """
    index: int
    unit_start: int
    unit_end: int
    context_before: int
    context_after: int

    def selectable(self, t: Transcript) -> list[Unit]:
        return t.units[self.unit_start:self.unit_end]

    def before(self, t: Transcript) -> list[Unit]:
        return t.units[self.context_before:self.unit_start]

    def after(self, t: Transcript) -> list[Unit]:
        return t.units[self.unit_end:self.context_after]

    def contains(self, unit_idx: int) -> bool:
        return self.unit_start <= unit_idx < self.unit_end


def build_windows(t: Transcript, *, window_seconds: float = 900.0, overlap_seconds: float = 120.0,
                  context_seconds: float = 60.0) -> list[Window]:
    """Overlapping windows over the units; a single window when the transcript is short.

    A trailing remainder shorter than 20 % of a window is absorbed into the previous
    window instead of becoming a tiny window of its own.
    """
    units = t.units
    n = len(units)
    if n == 0:
        return []
    starts = [u.start for u in units]
    ends = [u.end for u in units]
    min_tail = window_seconds * 0.2
    windows: list[Window] = []
    first = 0
    while True:
        limit = units[first].start + window_seconds
        end = max(bisect.bisect_right(ends, limit, lo=first), first + 1)   # exclusive
        if end >= n or units[-1].end - units[end].start < min_tail:
            end = n
        ctx_before = bisect.bisect_left(starts, units[first].start - context_seconds, hi=first)
        ctx_after = max(bisect.bisect_right(ends, units[end - 1].end + context_seconds, lo=end), end)
        windows.append(Window(index=len(windows), unit_start=first, unit_end=end,
                              context_before=ctx_before, context_after=ctx_after))
        if end >= n:
            return windows
        next_first = bisect.bisect_left(starts, units[end - 1].end - overlap_seconds, lo=first + 1)
        first = min(max(next_first, first + 1), end)


def candidates_per_window(clip_count: int) -> int:
    return max(4, math.ceil(clip_count * 1.5))


def _window_can_hold_clip(selectable: Sequence[Unit], options: ProjectOptions) -> bool:
    """False when even the whole selectable span (plus edge padding) is shorter than any valid clip."""
    if not selectable:
        return False
    lo, _ = duration_bounds(options)
    return selectable[-1].end - selectable[0].start + 2 * RANGE_PAD >= lo


def shortlist_size(clip_count: int) -> int:
    return min(max(3 * clip_count, 6), 18)


def duration_bounds(options: ProjectOptions) -> tuple[float, float]:
    """Accepted clip length: the requested range with a 15 % tolerance, never outside the editor's 3-180 s."""
    return (max(options.min_seconds * DURATION_LOW, MIN_CLIP_SECONDS),
            min(options.max_seconds * DURATION_HIGH, MAX_CLIP_SECONDS))


# ============================================================================ text helpers
_WS = re.compile(r"\s+")
_HASHTAG = re.compile(r"(?<!\w)#\w+")
_REPEAT_PUNCT = re.compile(r"([!?.])\1+")
_TITLE_PREFIX = re.compile(r"^(?:title|clip title)\s*[:\-]\s*", re.IGNORECASE)
_QUOTES = "\"'“”‘’«»`"
_ID_CHARS = re.compile(r"[^0-9A-Za-z_-]")


def _clip(text: str, limit: int) -> str:
    """Collapse whitespace and cut at a word boundary to at most ``limit`` characters."""
    text = _WS.sub(" ", text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0] if " " in text[:limit] else text[:limit]
    return cut.rstrip(" ,;:-–—")


def clean_title(text: str, *, max_chars: int = TITLE_MAX_CHARS) -> str:
    """Tidy a model title: no hashtags/emoji/wrapping quotes/shouty punctuation; ≤ max_chars."""
    text = unicodedata.normalize("NFC", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("So", "Cs", "Co", "Cc", "Cf"))
    text = _WS.sub(" ", text).strip()
    text = _TITLE_PREFIX.sub("", text)
    text = _HASHTAG.sub("", text)
    text = _REPEAT_PUNCT.sub(r"\1", text)
    text = _WS.sub(" ", text).strip()
    for _ in range(2):
        if len(text) >= 2 and text[0] in _QUOTES and text[-1] in _QUOTES:
            text = text[1:-1].strip()
    text = _calm_caps(text)
    text = _clip(text, max_chars)
    return text.rstrip(" ,;:-–—")


def _calm_caps(text: str) -> str:
    """Sentence-case a shouting ALL-CAPS title (short acronym-only titles are left alone)."""
    letters = [ch for ch in text if ch.isalpha()]
    if len(text.split()) < 3 or len(letters) < 12 or not all(ch.isupper() for ch in letters):
        return text
    lowered = text.lower()
    for i, ch in enumerate(lowered):
        if ch.isalpha():
            return lowered[:i] + ch.upper() + lowered[i + 1:]
    return lowered


def _fallback_title(units: Sequence[Unit]) -> str:
    return clean_title(" ".join(u.text for u in units[:3])) or "Untitled clip"


def _norm_unit_id(value: str) -> str:
    return _ID_CHARS.sub("", (value or "").strip()).lower()


def _id_lookup(t: Transcript) -> dict[str, int]:
    """Unit id -> index, also keyed by the normalised form (case, stray brackets/spaces)."""
    exact = unit_index(t)
    lookup = dict(exact)
    for uid, i in exact.items():
        lookup.setdefault(_norm_unit_id(uid), i)
    return lookup


def _find_unit(value: Any, lookup: dict[str, int]) -> int | None:
    """Index of the unit a model-supplied id names, or None when it names no unit."""
    text = str(value or "").strip()
    if not text:
        return None
    if text in lookup:
        return lookup[text]
    return lookup.get(_norm_unit_id(text))


def _norm_topic(topic: str) -> str:
    return _WS.sub(" ", (topic or "").lower()).strip()


def model_score_from(scores: CandidateScores) -> float:
    """0..100 from the four 1..5 sub-scores (completeness and hook weigh most)."""
    total = sum(weight * (getattr(scores, name) - 1) / 4.0 for name, weight in SCORE_WEIGHTS.items())
    return round(100.0 * total, 1)


def _clamp_int(value: Any, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return lo


def overlap_fraction(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Overlap as a fraction of the shorter range (0..1)."""
    inter = min(a_end, b_end) - max(a_start, b_start)
    shorter = min(a_end - a_start, b_end - b_start)
    if inter <= 0 or shorter <= 0:
        return 0.0
    return inter / shorter


def reason_code(reason: str) -> str:
    return reason.split(":", 1)[0]


# ============================================================================ validation
def _check_ids(item: CandidateOut, window: Window, t: Transcript,
               lookup: dict[str, int]) -> tuple[int, int, str | None]:
    """(start index, end index, rejection-or-None) for the model's unit IDs."""
    s, e = _find_unit(item.start_unit_id, lookup), _find_unit(item.end_unit_id, lookup)
    unknown = [_clip(raw, 24) or "empty" for raw, i in ((item.start_unit_id, s), (item.end_unit_id, e))
               if i is None]
    if unknown or s is None or e is None:
        return -1, -1, f"unknown_unit_id: {', '.join(unknown)}"
    outside = [t.units[i].id for i in (s, e) if not window.contains(i)]
    if outside:
        return s, e, f"outside_window: {', '.join(outside)}"
    if s > e:
        return s, e, f"reversed_range: {t.units[s].id} after {t.units[e].id}"
    return s, e, None


def invalid_id_details(rejected: Sequence[tuple[dict, str]], window: Window,
                       t: Transcript) -> tuple[list[str], list[tuple[str, str]]]:
    """From ID rejections: (ids that are unknown/outside the window, reversed (start, end) pairs)."""
    lookup = _id_lookup(t)
    bad_ids: list[str] = []
    reversed_pairs: list[tuple[str, str]] = []
    for raw, reason in rejected:
        code = reason_code(reason)
        if code not in ID_REJECTIONS:
            continue
        start_raw, end_raw = str(raw.get("start_unit_id", "")), str(raw.get("end_unit_id", ""))
        if code == "reversed_range":
            reversed_pairs.append((start_raw, end_raw))
            continue
        for value in (start_raw, end_raw):
            i = _find_unit(value, lookup)
            if i is None or not window.contains(i):
                bad_ids.append(value)
    return bad_ids, reversed_pairs


def _flag_rejection(item: CandidateOut) -> str | None:
    if item.is_ad:
        return "flagged_ad"
    if item.is_housekeeping:
        return "flagged_housekeeping"
    if item.needs_prior_context:
        return "needs_prior_context"
    return None


SENTENCE_SEARCH_UNITS = 8      # how far complete_sentences looks for a sentence edge


def complete_sentences(t: Transcript, s: int, e: int, lo: float, hi: float) -> tuple[int, int] | None:
    """Unit range ``[s, e]`` moved so the clip starts at a sentence start and ends at a sentence end.

    Units are split at long pauses, so a unit can start or end mid-sentence (e.g. "…as you'll see
    in future lectures," / pause / "crucial doctrines remain ambiguous."). Prefers extending outward
    (keeps the chosen moment whole) while the derived length stays within ``[lo, hi]``, then trimming
    inward to whole sentences. Returns None when no complete-sentence range fits.
    """
    units = t.units
    n = len(units)

    def starts(i: int) -> bool:
        return i == 0 or is_sentence_end(units[i - 1].text)

    def ends(i: int) -> bool:
        return i == n - 1 or is_sentence_end(units[i].text)

    def fits(a: int, b: int) -> bool:
        a_start, b_end = range_for_units(t, units[a].id, units[b].id)
        return lo <= b_end - a_start <= hi

    if starts(s) and ends(e):
        return s, e
    out_s = next((i for i in range(s, max(-1, s - SENTENCE_SEARCH_UNITS - 1), -1) if starts(i)), None)
    out_e = next((j for j in range(e, min(n, e + SENTENCE_SEARCH_UNITS + 1)) if ends(j)), None)
    in_s = next((i for i in range(s, e + 1) if starts(i)), None)
    in_e = next((j for j in range(e, s - 1, -1) if ends(j)), None)
    for a, b in ((out_s, out_e), (in_s, out_e), (out_s, in_e), (in_s, in_e)):
        if a is not None and b is not None and a <= b and fits(a, b):
            return a, b
    return None


def validate_candidates(raw: list[dict], window: Window, t: Transcript,
                        options: ProjectOptions) -> tuple[list[Candidate], list[tuple[dict, str]]]:
    """Turn raw model candidates into ``Candidate`` objects; return (accepted, [(raw, reason)]).

    Reasons are ``code`` or ``code: detail`` with codes: malformed, unknown_unit_id,
    outside_window, reversed_range, flagged_ad, flagged_housekeeping,
    needs_prior_context, too_short, too_long. Accepted candidates get provisional ids
    (``w00c00``); ``OpenAISelector.find_candidates`` renumbers them globally.
    """
    lookup = _id_lookup(t)
    lo, hi = duration_bounds(options)
    accepted: list[Candidate] = []
    rejected: list[tuple[dict, str]] = []
    for n, item_raw in enumerate(raw):
        try:
            item = CandidateOut.model_validate(item_raw)
        except pydantic.ValidationError:
            rejected.append((item_raw, "malformed"))
            continue
        s, e, problem = _check_ids(item, window, t, lookup)
        problem = problem or _flag_rejection(item)
        if problem:
            rejected.append((item_raw, problem))
            continue
        s, e = complete_sentences(t, s, e, lo, hi) or (s, e)
        start_id, end_id = t.units[s].id, t.units[e].id
        src_start, src_end = range_for_units(t, start_id, end_id)
        duration = src_end - src_start
        if duration < lo:
            rejected.append((item_raw, f"too_short: {duration:.1f}s < {lo:.1f}s"))
            continue
        if duration > hi:
            rejected.append((item_raw, f"too_long: {duration:.1f}s > {hi:.1f}s"))
            continue
        units = t.units[s:e + 1]
        scores = CandidateScores(**{k: _clamp_int(getattr(item.scores, k), 1, 5) for k in SCORE_WEIGHTS})
        accepted.append(Candidate(
            id=f"w{window.index:02d}c{n:02d}",
            window_index=window.index,
            start_unit_id=start_id,
            end_unit_id=end_id,
            src_start=src_start,
            src_end=src_end,
            title=clean_title(item.title) or _fallback_title(units),
            hook=_clip(item.hook, 200),
            payoff=_clip(item.payoff, 200),
            reason=_clip(item.reason, 300),
            topic=_clip(item.topic, 40) or "general",
            scores=scores,
            model_score=model_score_from(scores),
        ))
    return accepted, rejected


def _rank_key(c: Candidate) -> tuple[float, float, float]:
    return (-c.model_score, c.src_start, c.src_end)


def dedupe_candidates(c: list[Candidate], *, overlap_ratio: float = 0.5) -> list[Candidate]:
    """Drop candidates overlapping a better one by ≥ overlap_ratio of the shorter clip.

    The higher ``model_score`` wins (earlier start on ties). Result is in time order.
    """
    kept: list[Candidate] = []
    for cand in sorted(c, key=_rank_key):
        if all(overlap_fraction(cand.src_start, cand.src_end, k.src_start, k.src_end) < overlap_ratio
               for k in kept):
            kept.append(cand)
    return sorted(kept, key=lambda x: (x.src_start, x.src_end))


def shortlist(c: list[Candidate], options: ProjectOptions) -> list[Candidate]:
    """Top ``min(max(3*clip_count, 6), 18)`` by model_score; ties prefer a not-yet-seen topic."""
    limit = shortlist_size(options.clip_count)
    remaining = sorted(c, key=_rank_key)
    chosen: list[Candidate] = []
    topics: Counter[str] = Counter()
    while remaining and len(chosen) < limit:
        best = remaining[0].model_score
        tied = [x for x in remaining if math.isclose(x.model_score, best, abs_tol=1e-6)]
        pick = min(tied, key=lambda x: (topics[_norm_topic(x.topic)], x.src_start))
        chosen.append(pick)
        remaining.remove(pick)
        topics[_norm_topic(pick.topic)] += 1
    return chosen


def renumber_candidates(c: list[Candidate]) -> list[Candidate]:
    """Stable global ids c001, c002, ... in time order."""
    ordered = sorted(c, key=lambda x: (x.src_start, x.src_end))
    return [x.model_copy(update={"id": f"c{i:03d}"}) for i, x in enumerate(ordered, start=1)]


# ============================================================================ review helpers
def boundary_options(n_units: int, start_idx: int, end_idx: int,
                     max_shift: int = BOUNDARY_SHIFT_UNITS) -> tuple[list[int], list[int]]:
    """Unit indices the reviewer may choose as new start / end (original included)."""
    starts = list(range(max(0, start_idx - max_shift), min(end_idx, start_idx + max_shift) + 1))
    ends = list(range(max(start_idx, end_idx - max_shift), min(n_units - 1, end_idx + max_shift) + 1))
    return starts, ends


def _context_units(t: Transcript, start_idx: int, end_idx: int,
                   seconds: float = REVIEW_CONTEXT_SECONDS) -> tuple[list[Unit], list[Unit]]:
    units = t.units
    src_start, src_end = units[start_idx].start, units[end_idx].end
    i = start_idx
    while i > 0 and units[i - 1].end > src_start - seconds:
        i -= 1
    j = end_idx + 1
    while j < len(units) and units[j].start < src_end + seconds:
        j += 1
    return units[i:start_idx], units[end_idx + 1:j]


def _resolve_boundaries(c: Candidate, d: ReviewDecision, t: Transcript, lookup: dict[str, int],
                        bounds: tuple[float, float]) -> tuple[str, str, float, float] | None:
    """Apply the reviewer's boundary nudge if valid; else fall back to the candidate's own range.

    A nudge is valid when the new ids are in the candidate's allowed lists (±3 units),
    start ≤ end and the derived duration stays within the accepted bounds. When only one
    side of a nudge is valid, that side is applied alone. The candidate's own range is
    kept as validated, but never when it breaks the editor's 3-180 s limit (e.g. a
    candidate cached before that limit applied).
    """
    s0, e0 = lookup.get(c.start_unit_id), lookup.get(c.end_unit_id)
    if s0 is None or e0 is None or s0 > e0:
        return None
    allowed_starts, allowed_ends = boundary_options(len(t.units), s0, e0)
    new_s, new_e = _find_unit(d.start_unit_id, lookup), _find_unit(d.end_unit_id, lookup)
    options = [(new_s, new_e), (new_s, e0), (s0, new_e), (s0, e0)]
    lo, hi = bounds
    for s, e in options:
        if s is None or e is None or s > e:
            continue
        if (s, e) != (s0, e0) and (s not in allowed_starts or e not in allowed_ends):
            continue
        start_id, end_id = t.units[s].id, t.units[e].id
        src_start, src_end = range_for_units(t, start_id, end_id)
        length = src_end - src_start
        if lo <= length <= hi or ((s, e) == (s0, e0) and MIN_CLIP_SECONDS <= length <= MAX_CLIP_SECONDS):
            return start_id, end_id, src_start, src_end
    return None


def _sentence_complete_range(t: Transcript, lookup: dict[str, int], rng: tuple[str, str, float, float],
                             bounds: tuple[float, float]) -> tuple[str, str, float, float]:
    """Re-apply complete_sentences after the reviewer's boundary nudge (keeps rng if nothing fits)."""
    s, e = lookup.get(rng[0]), lookup.get(rng[1])
    if s is None or e is None:
        return rng
    snapped = complete_sentences(t, s, e, *bounds)
    if snapped is None or snapped == (s, e):
        return rng
    start_id, end_id = t.units[snapped[0]].id, t.units[snapped[1]].id
    return (start_id, end_id, *range_for_units(t, start_id, end_id))


def _default_models() -> tuple[str, str]:
    try:
        from ..config import get_settings
        s = get_settings()
        return s.select_model, s.review_model
    except Exception:  # pragma: no cover - settings unavailable
        return "unknown", "unknown"


def apply_review(cands: list[Candidate], decisions: list[ReviewDecision], t: Transcript, options: ProjectOptions,
                 *, min_score: float = 55.0, select_model: str | None = None,
                 review_model: str | None = None) -> list[ClipSelection]:
    """Final ranked clips: valid decisions only, ≥ min_score, deduped, ≤ clip_count, best first."""
    by_id = {c.id: c for c in cands}
    lookup = _id_lookup(t)
    bounds = duration_bounds(options)
    if select_model is None or review_model is None:
        default_select, default_review = _default_models()
        select_model = select_model or default_select
        review_model = review_model or default_review

    resolved: list[tuple[ReviewDecision, Candidate, tuple[str, str, float, float], float]] = []
    seen: set[str] = set()
    for d in decisions:
        c = by_id.get(d.candidate_id)
        if c is None or d.candidate_id in seen:
            continue
        seen.add(d.candidate_id)
        score = max(0.0, min(100.0, float(d.final_score)))
        if not d.keep or score < min_score:
            continue
        rng = _resolve_boundaries(c, d, t, lookup, bounds)
        if rng is not None:
            rng = _sentence_complete_range(t, lookup, rng, bounds)
            resolved.append((d, c, rng, score))

    resolved.sort(key=lambda r: (-r[3], -r[1].model_score, r[2][2]))
    chosen: list[tuple[ReviewDecision, Candidate, tuple[str, str, float, float], float]] = []
    claimed_duplicates: set[str] = set()    # candidates a kept decision said it duplicates
    for d, c, rng, score in resolved:
        dup = d.duplicate_of if d.duplicate_of in by_id and d.duplicate_of != c.id else None
        kept_ids = {x[1].id for x in chosen}
        if dup is not None and dup in kept_ids:
            continue                         # a better-scored twin is already kept
        if c.id in claimed_duplicates:
            continue                         # a better-scored candidate claimed to cover this one
        if any(overlap_fraction(rng[2], rng[3], x[2][2], x[2][3]) >= 0.5 for x in chosen):
            continue
        chosen.append((d, c, rng, score))
        if dup is not None:
            claimed_duplicates.add(dup)
        if len(chosen) >= options.clip_count:
            break

    version = f"{PROMPT_VERSION}+{REVIEW_PROMPT_VERSION}"
    return [
        ClipSelection(
            candidate_id=c.id, start_unit_id=rng[0], end_unit_id=rng[1], src_start=rng[2], src_end=rng[3],
            title=clean_title(d.title) or c.title, hook=c.hook,
            reason=_clip(d.reason, REVIEW_REASON_MAX_CHARS) or c.reason, topic=c.topic, score=round(score, 1),
            sub_scores={
                "opening": float(d.opening), "completeness": float(d.completeness),
                "interest": float(d.interest), "visual": float(d.visual), "model_score": c.model_score,
                "select_hook": float(c.scores.hook), "select_completeness": float(c.scores.completeness),
                "select_value": float(c.scores.value), "select_emotion": float(c.scores.emotion),
            },
            select_model=select_model, review_model=review_model, prompt_version=version,
        )
        for d, c, rng, score in chosen
    ]


def review_frame_times(c: Candidate, n: int = FRAMES_PER_CANDIDATE) -> list[float]:
    """Suggested source times for review frames (evenly inside the clip, away from the cuts)."""
    duration = c.src_end - c.src_start
    return [c.src_start + duration * (i + 1) / (n + 1) for i in range(n)]


def image_data_url(path: Path) -> str | None:
    """``data:`` URL for a JPEG/PNG frame, or None when unreadable/empty/too large."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data or len(data) > MAX_FRAME_BYTES:
        return None
    mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# ============================================================================ sub-request cache
M = TypeVar("M", bound=BaseModel)


def _transcript_fingerprint(t: Transcript) -> str:
    """Hash of everything a window's prompt and validation read from the transcript (units, word times)."""
    return stable_hash(t.duration, [(u.id, u.start, u.end, u.text, u.word_start, u.word_end) for u in t.units],
                       [(w.text, w.start, w.end) for w in t.words])


def _cached_list(cache: JsonCache | None, key: str, model: type[M]) -> list[M] | None:
    """A cached result list, or None on a miss (an unreadable entry counts as a miss and is re-requested)."""
    if cache is None:
        return None
    try:
        value = cache.get(key)
        if value is None:
            return None
        if not isinstance(value, list):
            raise TypeError(f"expected a list, got {type(value).__name__}")
        return [model.model_validate(item) for item in value]
    except Exception:  # noqa: BLE001 - a bad cache entry must never fail the job
        log.warning("ignoring an unreadable cached %s result", model.__name__, exc_info=True)
        return None


# ============================================================================ provider adapter
@dataclass
class _CallResult:
    parsed: BaseModel | None
    problem: str | None = None        # refusal | incomplete | invalid_output


def _attempt_key(key: str, max_output_tokens: int, corrections: Sequence[str]) -> str:
    """Key of one paid attempt inside a window/batch.

    Its corrections are derived only from the earlier attempts' answers, so a resume that
    replays those answers from the cache arrives at the same key for the next attempt.
    """
    return stable_hash(key, "attempt", max_output_tokens, len(corrections), list(corrections))


def _cached_attempt(cache: JsonCache, key: str, model: type[M]) -> _CallResult | None:
    """A stored attempt answer, or None on a miss (an unreadable entry counts as a miss)."""
    try:
        value = cache.get(key)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError(f"expected an object, got {type(value).__name__}")
        if value.get("parsed") is None:
            problem = value.get("problem")
            if not isinstance(problem, str) or problem not in _FAILURE_TEXT:
                raise ValueError(f"unknown problem {problem!r}")
            return _CallResult(None, problem)
        return _CallResult(model.model_validate(value["parsed"]))
    except Exception:  # noqa: BLE001 - a bad cache entry must never fail the job
        log.warning("ignoring an unreadable cached %s attempt", model.__name__, exc_info=True)
        return None


def _forget_attempts(cache: JsonCache | None, keys: Sequence[str]) -> None:
    """Clear the stored attempts of a window/batch that failed, so "Try again" asks afresh."""
    if cache is not None:
        for key in keys:
            cache.put(key, None)          # a None value reads back as a miss


@dataclass
class _ParseFailed:
    """Returned (not raised) when the SDK could not validate the JSON (usually truncation).

    Keeps the raw HTTP response: the request was answered and billed, so its request id,
    status and token usage are still recorded.
    """
    raw: Any

    @property
    def request_id(self) -> str | None:
        rid = getattr(self.raw, "request_id", None)
        return rid if isinstance(rid, str) and rid else None

    def body(self) -> dict[str, Any]:
        try:
            data = self.raw.http_response.json()
        except Exception:  # noqa: BLE001 - usage is best effort; the retry logic does not depend on it
            return {}
        return data if isinstance(data, dict) else {}


_FAILURE_TEXT = {
    "refusal": "the AI model declined to answer",
    "incomplete": "the AI model's answer was cut off before it finished",
    "invalid_output": "the AI model's answer could not be read",
    "invalid_ids": "the AI model kept naming transcript positions that do not exist",
    "missing_decisions": "the AI model did not return a decision for the candidates",
}


_WHAT = {"candidates": "clip selection", "review": "clip review"}


def _error_request_id(exc: BaseException) -> str | None:
    """OpenAI request id from the SDK exception chained under a ``ProviderError``."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        rid = getattr(current, "request_id", None)
        if isinstance(rid, str) and rid:
            return rid
        current = current.__cause__ or current.__context__
    return None


def _field(obj: Any, name: str) -> Any:
    """Attribute of an SDK object, or key of the same object as raw response JSON."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _usage_numbers(resp: Any) -> dict[str, int | None]:
    usage = _field(resp, "usage")
    if usage is None:
        return {}
    in_details = _field(usage, "input_tokens_details")
    out_details = _field(usage, "output_tokens_details")
    return {
        "input_tokens": _field(usage, "input_tokens"),
        "output_tokens": _field(usage, "output_tokens"),
        "cached_tokens": _field(in_details, "cached_tokens") if in_details is not None else None,
        "reasoning_tokens": _field(out_details, "reasoning_tokens") if out_details is not None else None,
    }


def _refusal(resp: Any) -> str | None:
    for item in _field(resp, "output") or []:
        if _field(item, "type") != "message":
            continue
        for part in _field(item, "content") or []:
            if _field(part, "type") == "refusal":
                return str(_field(part, "refusal") or "refused")
    return None


class OpenAISelector:
    """``ClipSelector`` backed by the OpenAI Responses API with strict Structured Outputs."""

    provider = PROVIDER

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self.select_model = settings.select_model
        self.review_model = settings.review_model
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_client(self.settings)
        return self._client

    # ------------------------------------------------------------------ candidates
    def find_candidates(self, transcript: Transcript, options: ProjectOptions, *, cancel: CancelToken,
                        usage: UsageSink, cache: JsonCache | None = None) -> list[Candidate]:
        # Overlap must exceed the longest acceptable clip so no clip straddles a seam unseen.
        overlap = max(120.0, duration_bounds(options)[1] + 10.0)
        windows = build_windows(transcript, overlap_seconds=overlap)
        transcript_key = _transcript_fingerprint(transcript) if cache is not None else ""
        found: list[Candidate] = []
        for window in windows:
            cancel.raise_if_cancelled()      # honoured only after the previous window's result was cached
            key = self._window_key(transcript_key, window, len(windows), options) if cache is not None else ""
            mined = _cached_list(cache, key, Candidate)
            if mined is None:
                mined = self._mine_window(transcript, window, len(windows), options, cancel=cancel, usage=usage,
                                          cache=cache, key=key)
                if cache is not None:
                    cache.put(key, [c.model_dump(mode="json") for c in mined])
            found.extend(mined)
        return renumber_candidates(dedupe_candidates(found))

    def _window_key(self, transcript_key: str, window: Window, window_count: int, options: ProjectOptions) -> str:
        """Every input of one window's request (and of validating its answer)."""
        return stable_hash("candidates-window", PROMPT_VERSION, self.select_model,
                           self.settings.select_reasoning_effort, transcript_key,
                           [window.index, window.unit_start, window.unit_end, window.context_before,
                            window.context_after], window_count, options.min_seconds, options.max_seconds,
                           options.clip_count, duration_bounds(options))

    def _mine_window(self, t: Transcript, window: Window, window_count: int, options: ProjectOptions, *,
                     cancel: CancelToken, usage: UsageSink, cache: JsonCache | None = None,
                     key: str = "") -> list[Candidate]:
        """One window: request, validate, and retry (≤ 2 extra) on refusal/truncation/bad IDs.

        Valid candidates from an attempt that is retried because of bad IDs are kept and
        merged with the retry's answer, so a correction never loses good moments. Each paid
        attempt is checkpointed in ``cache`` (under the window ``key``) as soon as it returns.
        """
        selectable = window.selectable(t)
        if not _window_can_hold_clip(selectable, options):
            log.info("candidates window %d: too little speech for a %.0f s clip; skipped",
                     window.index, options.min_seconds)
            return []
        max_items = candidates_per_window(options.clip_count)
        max_tokens = self.settings.select_max_output_tokens
        corrections: list[str] = []
        attempt_keys: list[str] = []
        pool: list[Candidate] = []
        problem = "invalid_output"
        for _ in range(MAX_ATTEMPTS):
            messages = prompts.build_candidate_input(
                before=window.before(t), selectable=selectable, after=window.after(t), options=options,
                window_number=window.index + 1, window_count=window_count, max_candidates=max_items,
                corrections=corrections)
            attempt_keys.append(_attempt_key(key, max_tokens, corrections))
            result = self._checkpointed(cache, attempt_keys[-1], CandidateListOut, lambda: self._structured_call(
                stage="candidates", model=self.select_model, effort=self.settings.select_reasoning_effort,
                max_output_tokens=max_tokens, messages=messages, text_format=CandidateListOut,
                cancel=cancel, usage=usage))
            if not isinstance(result.parsed, CandidateListOut):
                problem = result.problem or "invalid_output"
                corrections.append(self._correction(problem, max_items))
                continue
            raw = [c.model_dump() for c in result.parsed.candidates]
            accepted, rejected = validate_candidates(raw, window, t, options)
            self._log_rejections("candidates", window.index, rejected)
            pool = dedupe_candidates(pool + accepted)
            bad = [r for r in rejected if reason_code(r[1]) in ID_REJECTIONS]
            if bad and len(bad) * 2 >= len(raw):
                problem = "invalid_ids"
                bad_ids, reversed_pairs = invalid_id_details(bad, window, t)
                corrections.append(prompts.invalid_ids_correction(
                    bad_ids, selectable[0].id, selectable[-1].id, reversed_pairs))
                continue
            return sorted(pool, key=_rank_key)[:max_items]
        if pool:
            return sorted(pool, key=_rank_key)[:max_items]
        _forget_attempts(cache, attempt_keys)
        raise self._failure("Finding clip moments", problem, tokens_setting="CLIPPER_SELECT_MAX_OUTPUT_TOKENS")

    # ------------------------------------------------------------------ review
    def review(self, transcript: Transcript, candidates: list[Candidate],
               frames: dict[str, list[tuple[float, Path]]], options: ProjectOptions, *,
               cancel: CancelToken, usage: UsageSink, cache: JsonCache | None = None) -> list[ReviewDecision]:
        idx = unit_index(transcript)
        items: list[tuple[Candidate, prompts.ReviewItem]] = []
        for c in candidates:
            item = self._review_item(transcript, idx, c, frames.get(c.id, []))
            if item is not None:
                items.append((c, item))
        decisions: list[ReviewDecision] = []
        for i in range(0, len(items), REVIEW_BATCH_SIZE):
            cancel.raise_if_cancelled()      # honoured only after the previous batch's result was cached
            batch = items[i:i + REVIEW_BATCH_SIZE]
            key = self._batch_key(batch, options) if cache is not None else ""
            reviewed = _cached_list(cache, key, ReviewDecision)
            if reviewed is None:
                reviewed = self._review_batch(batch, options, cancel=cancel, usage=usage, cache=cache, key=key)
                if cache is not None:
                    cache.put(key, [d.model_dump(mode="json") for d in reviewed])
            decisions.extend(reviewed)
        return decisions

    def _batch_key(self, batch: list[tuple[Candidate, prompts.ReviewItem]], options: ProjectOptions) -> str:
        """Every input of one review request: candidates, ranges, texts, frames (by content) and limits."""
        parts = [[c.model_dump(mode="json"), prompts.review_candidate_text(item),
                  [(src_time, hashlib.sha256(url.encode("ascii")).hexdigest()) for src_time, url in item.frames]]
                 for c, item in batch]
        return stable_hash("review-batch", REVIEW_PROMPT_VERSION, self.review_model,
                           self.settings.review_reasoning_effort, parts, options.min_seconds, options.max_seconds,
                           options.clip_count, duration_bounds(options))

    def _review_item(self, t: Transcript, idx: dict[str, int], c: Candidate,
                     frames: Sequence[tuple[float, Path]]) -> prompts.ReviewItem | None:
        s, e = idx.get(c.start_unit_id), idx.get(c.end_unit_id)
        if s is None or e is None or s > e:
            log.warning("review: candidate %s no longer matches the transcript; skipped", c.id)
            return None
        starts, ends = boundary_options(len(t.units), s, e)
        before, after = _context_units(t, s, e)
        images: list[tuple[float, str]] = []
        for src_time, path in list(frames)[:FRAMES_PER_CANDIDATE]:
            url = image_data_url(Path(path))
            if url is not None:
                images.append((float(src_time), url))
        return prompts.ReviewItem(
            candidate_id=c.id, src_start=c.src_start, src_end=c.src_end, draft_title=c.title, draft_hook=c.hook,
            units=t.units[s:e + 1], context_before=before, context_after=after,
            start_options=[t.units[i] for i in starts], end_options=[t.units[i] for i in ends],
            original_start_unit_id=c.start_unit_id, original_end_unit_id=c.end_unit_id, frames=images)

    def _review_batch(self, batch: list[tuple[Candidate, prompts.ReviewItem]], options: ProjectOptions, *,
                      cancel: CancelToken, usage: UsageSink, cache: JsonCache | None = None, key: str = "",
                      follow_up: bool = True) -> list[ReviewDecision]:
        """One batch: request, convert, and retry (≤ 2 extra) on refusal/truncation/mostly missing decisions.

        Every candidate needs its own decision, so a truncated answer is asked again for all of
        them (never "fewer items"), and candidates still without a decision are then reviewed
        once more on their own (``follow_up``). Each paid attempt is checkpointed in ``cache``.
        """
        by_id = {c.id: c for c, _ in batch}
        expected = list(by_id)
        max_tokens = self.settings.review_max_output_tokens
        corrections: list[str] = []
        attempt_keys: list[str] = []
        best: list[ReviewDecision] = []
        decisions: list[ReviewDecision] | None = None
        problem = "invalid_output"
        for _ in range(MAX_ATTEMPTS):
            messages = prompts.build_review_input([item for _, item in batch], options, corrections)
            attempt_keys.append(_attempt_key(key, max_tokens, corrections))
            result = self._checkpointed(cache, attempt_keys[-1], ReviewListOut, lambda: self._structured_call(
                stage="review", model=self.review_model, effort=self.settings.review_reasoning_effort,
                max_output_tokens=max_tokens, messages=messages, text_format=ReviewListOut,
                cancel=cancel, usage=usage))
            if not isinstance(result.parsed, ReviewListOut):
                problem = result.problem or "invalid_output"
                corrections.append(prompts.review_truncated_correction(len(batch)) if problem == "incomplete"
                                   else self._correction(problem, len(batch)))
                continue
            found, unknown = self._convert_decisions(result.parsed.decisions, by_id)
            missing = [cid for cid in expected if cid not in {d.candidate_id for d in found}]
            if missing and len(missing) * 2 > len(expected):
                problem = "missing_decisions"
                if len(found) > len(best):
                    best = found
                corrections.append(prompts.missing_decisions_correction(missing, unknown))
                continue
            decisions = found
            break
        if decisions is None:
            if not best:
                _forget_attempts(cache, attempt_keys)
                raise self._failure("Reviewing clips", problem, tokens_setting="CLIPPER_REVIEW_MAX_OUTPUT_TOKENS")
            decisions = best
        return self._review_missing(batch, decisions, options, cancel=cancel, usage=usage, cache=cache, key=key,
                                    follow_up=follow_up)

    def _review_missing(self, batch: list[tuple[Candidate, prompts.ReviewItem]], decisions: list[ReviewDecision],
                        options: ProjectOptions, *, cancel: CancelToken, usage: UsageSink,
                        cache: JsonCache | None, key: str, follow_up: bool) -> list[ReviewDecision]:
        """Ask once more, as their own request, about candidates the batch answer left without a decision."""
        decided = {d.candidate_id for d in decisions}
        rest = [(c, item) for c, item in batch if c.id not in decided]
        if not rest or not follow_up:
            return decisions             # a follow-up's own gaps are reported by the batch that asked for it
        try:
            sub_key = stable_hash(key, "missing", [c.id for c, _ in rest]) if cache is not None else ""
            more = self._review_batch(rest, options, cancel=cancel, usage=usage, cache=cache, key=sub_key,
                                      follow_up=False)
        except ProviderError as exc:
            if exc.code != "provider_invalid_output":
                raise
            more = []                    # the model would not decide on them: keep the decisions it gave
        order = {c.id: i for i, (c, _) in enumerate(batch)}
        decisions = sorted(decisions + more, key=lambda d: order[d.candidate_id])
        if len(decisions) < len(batch):
            log.warning("review: no decision for %d of %d candidates", len(batch) - len(decisions), len(batch))
        return decisions

    @staticmethod
    def _convert_decisions(raw: Sequence[ReviewDecisionOut],
                           by_id: dict[str, Candidate]) -> tuple[list[ReviewDecision], list[str]]:
        decisions: list[ReviewDecision] = []
        unknown: list[str] = []
        seen: set[str] = set()
        for out in raw:
            cid = out.candidate_id.strip()
            cand = by_id.get(cid)
            if cand is None:
                unknown.append(cid)
                continue
            if cid in seen:
                continue
            seen.add(cid)
            dup = (out.duplicate_of or "").strip() or None
            if dup not in by_id or dup == cid:
                dup = None
            decisions.append(ReviewDecision(
                candidate_id=cid, keep=bool(out.keep), final_score=float(_clamp_int(out.final_score, 0, 100)),
                title=clean_title(out.title) or cand.title,
                reason=_clip(out.reason, REVIEW_REASON_MAX_CHARS) or cand.reason,
                opening=_clamp_int(out.opening, 1, 5), completeness=_clamp_int(out.completeness, 1, 5),
                interest=_clamp_int(out.interest, 1, 5), visual=_clamp_int(out.visual, 1, 5),
                duplicate_of=dup,
                start_unit_id=_norm_unit_id(out.start_unit_id) or cand.start_unit_id,
                end_unit_id=_norm_unit_id(out.end_unit_id) or cand.end_unit_id,
            ))
        return decisions, unknown

    # ------------------------------------------------------------------ one request
    @staticmethod
    def _checkpointed(cache: JsonCache | None, key: str, text_format: type[M],
                      send: Callable[[], _CallResult]) -> _CallResult:
        """One attempt: replay its stored answer, or pay for it and store the answer at once.

        The answer (parsed or a refusal/truncation/invalid problem) is written before the
        caller's next attempt checks for a cancel, so a resume never re-bills it.
        """
        if cache is None:
            return send()
        stored = _cached_attempt(cache, key, text_format)
        if stored is not None:
            return stored
        result = send()
        cache.put(key, {"parsed": result.parsed.model_dump(mode="json") if result.parsed is not None else None,
                        "problem": result.problem})
        return result

    def _structured_call(self, *, stage: str, model: str, effort: str, max_output_tokens: int,
                         messages: list[dict[str, Any]], text_format: type[T], cancel: CancelToken,
                         usage: UsageSink) -> _CallResult:
        cancel.raise_if_cancelled()
        kwargs: dict[str, Any] = {"model": model, "input": messages, "text_format": text_format,
                                  "max_output_tokens": max_output_tokens, "store": False}
        if effort:
            kwargs["reasoning"] = {"effort": effort}     # never combined with temperature/top_p
        client = self.client

        def _send() -> Any:
            # The raw response survives a failed parse: cut-off JSON makes the SDK's Structured
            # Outputs post-parser raise pydantic.ValidationError after the request was billed.
            raw = client.responses.with_raw_response.parse(**kwargs)
            try:
                return raw.parse()
            except pydantic.ValidationError:
                return _ParseFailed(raw)

        started = time.monotonic()

        def record(**kw: Any) -> None:
            usage.record(stage=stage, provider=PROVIDER, model=model,
                         duration_s=round(time.monotonic() - started, 3), **kw)

        try:
            resp = call_openai(_send, what=_WHAT[stage], cancel=cancel, model=model,
                               attempts=max(1, int(self.settings.openai_max_retries) + 1))
        except (CancelledError, InterruptedError_):
            raise
        except ClipperError as exc:
            record(request_id=_error_request_id(exc), status="error", error_code=exc.code)
            raise
        if isinstance(resp, _ParseFailed):
            body = resp.body()
            request_id, tokens = resp.request_id, _usage_numbers(body)
            if _field(body, "status") != "incomplete":
                record(request_id=request_id, status="error", error_code="invalid_output", **tokens)
                log.warning("%s: structured output failed validation (request %s)", stage, request_id)
                return _CallResult(None, "invalid_output")
            resp = body                      # JSON cut off mid-answer: handled as the truncation it is
        else:
            request_id = getattr(resp, "_request_id", None)
            tokens = _usage_numbers(resp)
        if _refusal(resp) is not None:
            record(request_id=request_id, status="error", error_code="refusal", **tokens)
            log.warning("%s: model refused (request %s)", stage, request_id)
            return _CallResult(None, "refusal")
        if _field(resp, "status") == "incomplete":
            details = _field(resp, "incomplete_details")
            why = (_field(details, "reason") if details is not None else None) or "unknown"
            record(request_id=request_id, status="error", error_code=f"incomplete_{why}", **tokens)
            log.warning("%s: response incomplete (%s, request %s)", stage, why, request_id)
            return _CallResult(None, "incomplete")
        parsed = getattr(resp, "output_parsed", None)
        if not isinstance(parsed, text_format):
            record(request_id=request_id, status="error", error_code="invalid_output", **tokens)
            log.warning("%s: no parsed output (request %s)", stage, request_id)
            return _CallResult(None, "invalid_output")
        record(request_id=request_id, status="ok", **tokens)
        return _CallResult(parsed)

    @staticmethod
    def _correction(problem: str, max_items: int) -> str:
        if problem == "refusal":
            return prompts.refusal_correction()
        if problem == "incomplete":
            return prompts.truncated_correction(max(2, max_items // 2))
        return prompts.invalid_output_correction(max_items)

    @staticmethod
    def _failure(what: str, problem: str, *, tokens_setting: str) -> ProviderError:
        cause = _FAILURE_TEXT.get(problem, _FAILURE_TEXT["invalid_output"])
        message = f"{what} failed after {MAX_ATTEMPTS} attempts: {cause}."
        if problem == "incomplete":
            message += f" If this keeps happening, raise {tokens_setting} in the .env file."
        else:
            message += " Please try again later."
        return ProviderError(message, code="provider_invalid_output", detail=f"{what}: {problem}")

    @staticmethod
    def _log_rejections(stage: str, window_index: int, rejected: list[tuple[dict, str]]) -> None:
        if rejected:
            counts = Counter(reason_code(r) for _, r in rejected)
            log.info("%s window %d: rejected %s", stage, window_index, dict(counts))
