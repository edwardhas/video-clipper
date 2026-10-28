"""Pure transcript normalisation: chunk merging, punctuation alignment, units, validation.

Everything here is deterministic and free of I/O. Times on the returned objects are
**source seconds** (``src``); provider results arrive in chunk-relative seconds and
are offset by ``ChunkPlan.start`` exactly once, in :func:`merge_chunks`.

Rules (see ``docs/ARCHITECTURE.md``):

* A word belongs to the chunk whose keep range ``[keep_start, keep_end)`` contains
  its midpoint; the same rule decides which chunk's copy of a segment is kept.
  Two chunks time a word straddling their seam independently, so at each seam:
  copies both chunks kept are one word when they are the same occurrence (k-th copy of
  that word in both chunks' shared audio) or, when that order is unclear, when they
  overlap/nearly coincide or each chunk heard that word only once there
  (``duplicate_words_dropped``); a word each chunk left to the other is kept from the
  earlier chunk (``seam_words_recovered``).
* Silence hallucinations: given the detected pauses (``silences``, source seconds), a word
  whose whole span lies inside one pause *and* whose provider segment reports
  ``no_speech_prob >= 0.5`` is dropped (``dropped_in_silence``); quiet real speech in a
  pause (low ``no_speech_prob``) and confident-silence segments over audible speech are kept.
* Word timings are never invented: they are only offset, clamped to the media and
  nudged to be monotonic/non-overlapping, and every such fix is counted in
  ``Transcript.fixes``.
* Display text (punctuation, casing) is aligned from segment text onto the raw word
  tokens; the raw token is kept in ``Word.raw``.
"""

from __future__ import annotations

import bisect
import difflib
import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, Sequence

from ..models import TRANSCRIPT_VERSION, Segment, Transcript, TranscriptChunk, Unit, Word
from .interfaces import RawChunkResult

if TYPE_CHECKING:  # pragma: no cover
    from ..media.audio import ChunkPlan

# ----------------------------------------------------------------------------- constants
MIN_WORDS = 80                 # fewer words than this → not enough speech for clips
MIN_COVERAGE = 0.10            # speech must cover at least 10 % of the media
MIN_WORDS_PER_MINUTE = 30.0    # averaged over the whole media
SPEECH_GAP_BRIDGE = 0.5        # pauses up to this long count as speech time in speech_stats
DUPLICATE_MAX_WORDS = 6        # longest run of words de-duplicated at a chunk seam
DUPLICATE_MID_TOLERANCE = 0.1  # seconds; non-overlapping copies of a word at a seam this close are one word
SEAM_MATCH_SECONDS = 1.0       # farthest apart (midpoints) two chunks' copies of one word may be timed
SEAM_HEARD_MARGIN = 0.5        # seconds around two seam copies searched for further copies of the word
SEAM_ORDER_TOLERANCE = 0.3     # seconds; two chunks' copies paired by occurrence order must be timed this close
SILENCE_TOLERANCE = 0.05       # seconds a word may reach past a detected pause and still lie inside it
SILENCE_NO_SPEECH_PROB = 0.5   # a word inside a pause is dropped when its segment's no_speech_prob is >= this
BOUNDARY_EDGE = 1e-3           # seconds; range_for_units edge nudge around zero-length boundary words
_EPS = 1e-6

TARGET_NOTE = (
    "The first release targets speech-led videos such as interviews, podcasts, talks and commentary."
)

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'"})
# Pure-punctuation tokens in segment text that attach to a neighbouring word ("Hello ...", "well —").
_JOINABLE = set(".,!?;:…-–—\"'“”‘’«»()[]{}¿¡")
_CLOSERS = "\"'”’»)]}"
_CLAUSE_END = (",", ";", ":", "—", "–")
_ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "e.g.", "i.e.", "etc.", "u.s.", "a.m.", "p.m.",
})
_DOTTED_ACRONYM = re.compile(r"(?:[A-Za-z]\.){2,}")
_INITIAL = re.compile(r"[A-HJ-Z]\.")      # "J." in "J. R. Tolkien"; not the pronoun "I." ending a sentence


def word_ids_for(starts: list[float]) -> list[str]:
    """Stable word ids derived from source start times in milliseconds (``w000012345``).

    Ids do not depend on a word's position, so re-merging a transcript (e.g. after a dropped
    hallucination) never shifts the ids that the owner's caption corrections refer to. Words starting
    in the same millisecond get ``-2``, ``-3`` … suffixes; ids sort lexically in time order.
    """
    ids: list[str] = []
    seen: dict[str, int] = {}
    for start in starts:
        base = f"w{max(0, int(round(start * 1000))):09d}"
        k = seen.get(base, 0) + 1
        seen[base] = k
        ids.append(base if k == 1 else f"{base}-{k}")
    return ids


def segment_id(n: int) -> str:
    return f"s{n:05d}"


def unit_id(n: int) -> str:
    return f"u{n:04d}"


# ----------------------------------------------------------------------------- tokens
def _fold(text: str) -> str:
    """Case/apostrophe-insensitive form used to compare two spellings of one token."""
    return unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).casefold()


def normalize_token(token: str) -> str:
    """Matching key: lowercase letters/digits only (``"Don't,"`` → ``"dont"``, ``"1,000"`` → ``"1000"``).

    Symbol-only tokens (``"&"``, ``"%"``) keep their symbols so they can still match each other.
    """
    folded = _fold(token).strip()
    core = "".join(ch for ch in folded if ch.isalnum())
    return core or folded


def _split_edges(token: str) -> tuple[str, str, str]:
    """``'"Hello,'`` → ``('"', 'Hello', ',')``; a token without letters/digits is all lead."""
    n = len(token)
    i = 0
    while i < n and not token[i].isalnum():
        i += 1
    if i == n:
        return token, "", ""
    j = n
    while j > i and not token[j - 1].isalnum():
        j -= 1
    return token[:i], token[i:j], token[j:]


def _with_case(raw_core: str, seg_core: str) -> str:
    """Capitalise the raw core's first letter when the segment capitalised it (sentence starts)."""
    if raw_core and seg_core and seg_core[0].isupper() and raw_core[0].islower():
        return raw_core[0].upper() + raw_core[1:]
    return raw_core


def _apply_edges(raw: str, *, lead: str = "", trail: str = "", case_from: str = "") -> str:
    r_lead, r_core, r_trail = _split_edges(raw)
    if not r_core:
        return raw
    core = _with_case(r_core, case_from) if case_from else r_core
    return (lead or r_lead) + core + (trail or r_trail)


def _merge_pair(raw: str, seg: str) -> str:
    """Display text for a raw word matched one-to-one with a segment token."""
    s_lead, s_core, s_trail = _split_edges(seg)
    r_lead, r_core, r_trail = _split_edges(raw)
    if not s_core or not r_core:
        return seg if _fold(seg) == _fold(raw) else raw
    if _fold(s_core) == _fold(r_core):
        return seg                       # same word: the segment's casing/punctuation wins
    # Same letters, different internal spelling ("1,000" vs "1000", "dont" vs "don't"):
    # keep the raw word and take only the unambiguous edge punctuation and capitalisation.
    return (s_lead or r_lead) + _with_case(r_core, s_core) + (s_trail or r_trail)


def _segment_tokens(segment_text: str) -> list[str]:
    """Whitespace tokens with pure-punctuation tokens glued to their neighbour."""
    tokens: list[str] = []
    pending_lead = ""
    for tok in segment_text.split():
        if all(ch in _JOINABLE for ch in tok):
            if tokens:
                tokens[-1] += tok
            else:
                pending_lead += tok
            continue
        tokens.append(pending_lead + tok)
        pending_lead = ""
    if pending_lead and tokens:
        tokens[-1] += pending_lead
    return tokens


def align_punctuation(raw_words: list[str], segment_text: str) -> list[str]:
    """Return display text for each raw word, taking punctuation/casing from ``segment_text``.

    The output has exactly one entry per input word (timings are untouched by design).
    Tokens are matched with :class:`difflib.SequenceMatcher` on :func:`normalize_token`
    keys. Same-letter re-tokenisations ("well-known" vs "well known") keep the raw
    words and only move the outer punctuation; equal-count mismatches are paired
    positionally; anything else stays raw, except that the segment's first/last
    punctuation is applied when the mismatch is at the very start/end.
    """
    raw = [w.strip() for w in raw_words]
    out = list(raw)
    seg_tokens = _segment_tokens(segment_text or "")
    raw_pos = [i for i, w in enumerate(raw) if normalize_token(w)]
    if not seg_tokens or not raw_pos:
        return out
    a = [normalize_token(raw[i]) for i in raw_pos]
    b = [normalize_token(t) for t in seg_tokens]
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                ri = raw_pos[i1 + k]
                out[ri] = _merge_pair(raw[ri], seg_tokens[j1 + k])
        elif tag == "replace":
            _align_replace(raw, out, raw_pos, seg_tokens, a, b, (i1, i2), (j1, j2))
        # "delete": extra raw words stay raw; "insert": extra segment tokens are ignored.
    return out


def _same_letter_group(a: list[str], b: list[str], i: int, i_end: int, j: int,
                       j_end: int) -> tuple[int, int] | None:
    """Smallest ``(i2, j2)`` with ``"".join(a[i:i2]) == "".join(b[j:j2])`` (re-tokenised words), if any."""
    left, right = a[i], b[j]
    i2, j2 = i + 1, j + 1
    while left != right:
        if right.startswith(left) and i2 < i_end:
            left += a[i2]
            i2 += 1
        elif left.startswith(right) and j2 < j_end:
            right += b[j2]
            j2 += 1
        else:
            return None
    return i2, j2


def _apply_group(raw: list[str], out: list[str], positions: list[int], toks: list[str]) -> None:
    """Raw words spelling the same letters as ``toks``: keep them, move only outer punctuation/case."""
    first_lead, first_core, _ = _split_edges(toks[0])
    _, _, last_trail = _split_edges(toks[-1])
    first, last = positions[0], positions[-1]
    if len(positions) == 1 and len(toks) == 1:
        out[first] = _merge_pair(raw[first], toks[0])
    elif first == last:
        out[first] = _apply_edges(raw[first], lead=first_lead, trail=last_trail, case_from=first_core)
    else:
        out[first] = _apply_edges(raw[first], lead=first_lead, case_from=first_core)
        out[last] = _apply_edges(raw[last], trail=last_trail)


def _align_replace(raw: list[str], out: list[str], raw_pos: list[int], toks: list[str], a: list[str],
                   b: list[str], span_a: tuple[int, int], span_b: tuple[int, int]) -> None:
    """Align a block where raw keys ``a[span_a]`` differ from segment keys ``b[span_b]``."""
    i, i_end = span_a
    j, j_end = span_b
    while i < i_end and j < j_end:          # leading same-letter groups ("well known" ~ "well-known,")
        group = _same_letter_group(a, b, i, i_end, j, j_end)
        if group is None:
            break
        _apply_group(raw, out, raw_pos[i:group[0]], toks[j:group[1]])
        i, j = group
    if i >= i_end or j >= j_end:
        return
    positions = raw_pos[i:i_end]
    rest = toks[j:j_end]
    if len(positions) == len(rest):       # substitutions ("colour" ~ "Color"): pair positionally
        for pos, tok in zip(positions, rest):
            out[pos] = _merge_pair(raw[pos], tok)
        return
    # Ambiguous: only the segment's very first/last punctuation is safe to place.
    first_lead, first_core, _ = _split_edges(rest[0])
    _, _, last_trail = _split_edges(rest[-1])
    if i == 0 and j == 0:
        out[positions[0]] = _apply_edges(raw[positions[0]], lead=first_lead, case_from=first_core)
    if i_end == len(a) and j_end == len(b):
        out[positions[-1]] = _apply_edges(out[positions[-1]], trail=last_trail)


# ----------------------------------------------------------------------------- units
def is_sentence_end(text: str) -> bool:
    """True when ``text`` ends a sentence (``. ? ! …`` plus closing quotes), ignoring abbreviations."""
    core = text.strip().rstrip(_CLOSERS)
    if not core:
        return False
    if core[-1] in "?!…":
        return True
    if core[-1] != ".":
        return False
    word = core.lstrip("\"'“‘«([{")
    if word.casefold() in _ABBREVIATIONS or _INITIAL.fullmatch(word) or _DOTTED_ACRONYM.fullmatch(word):
        return False
    return True


def _split_long(a: int, b: int, words: Sequence[Word], max_words: int) -> list[tuple[int, int]]:
    """Split the inclusive span ``[a, b]`` into pieces of at most ``max_words`` words."""
    spans: list[tuple[int, int]] = []
    min_first = max(1, max_words // 4)
    while b - a + 1 > max_words:
        cut = None
        for k in range(a + max_words - 1, a + min_first - 2, -1):
            if words[k].text.rstrip(_CLOSERS).endswith(_CLAUSE_END):
                cut = k
                break
        if cut is None:
            cut = a + max_words - 1
        spans.append((a, cut))
        a = cut + 1
    spans.append((a, b))
    return spans


def build_units(words: list[Word], *, max_words: int = 45, pause_break: float = 1.2) -> list[Unit]:
    """Group words into sentence-like units (IDs ``u0001``… in time order).

    A unit ends at sentence-final punctuation (not after abbreviations or initials), at
    a pause longer than ``pause_break`` seconds, or after ``max_words`` words — then at
    the last comma within the unit when there is one.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(words)
    for i, w in enumerate(words):
        last = i == n - 1
        if last or is_sentence_end(w.text) or words[i + 1].start - w.end > pause_break:
            spans.extend(_split_long(start, i, words, max_words))
            start = i + 1
    units: list[Unit] = []
    for k, (a, b) in enumerate(spans, start=1):
        text = " ".join(w.text for w in words[a:b + 1] if w.text)
        units.append(Unit(id=unit_id(k), start=words[a].start, end=words[b].end, text=text,
                          word_start=a, word_end=b))
    return units


# ----------------------------------------------------------------------------- merging
class _PlanLike(Protocol):
    index: int
    start: float
    end: float
    keep_start: float
    keep_end: float


@dataclass(eq=False)
class _Seg:
    chunk_index: int
    chunk_pos: int
    local_idx: int
    text: str
    start: float
    end: float
    keep_start: float
    keep_end: float
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    mid_kept: bool = False
    synthetic: bool = False
    silent: bool = False              # whole span inside a detected pause and no_speech_prob is high
    n_local_words: int = 0
    modified: bool = False
    members: list[int] = field(default_factory=list)


@dataclass(eq=False)
class _W:
    raw: str
    text: str
    start: float
    end: float
    chunk_index: int
    chunk_pos: int
    seg: _Seg | None


@dataclass(eq=False)
class _Group:
    """One chunk's words: the ones it owns (``kept``) plus what it heard around its keep range."""
    plan: _PlanLike
    pos: int
    n_segments: int
    segs: list[_Seg]
    kept: list[_W] = field(default_factory=list)
    heard: list[_W] = field(default_factory=list)    # every non-empty word the chunk returned
    early: list[_W] = field(default_factory=list)    # not owned: midpoint just before keep_start
    late: list[_W] = field(default_factory=list)     # not owned: midpoint at/just after keep_end


def _finite(*values: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


class _Silences:
    """Detected pauses (source seconds), sorted and merged, for "does this span lie inside one?"."""

    def __init__(self, silences: Sequence[tuple[float, float]] | None) -> None:
        merged: list[tuple[float, float]] = []
        for start, end in sorted((float(s), float(e)) for s, e in silences or () if _finite(s, e) and e > s):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        self.spans = merged
        self.starts = [s for s, _ in merged]

    def contains(self, start: float, end: float) -> bool:
        """True when ``[start, end]`` lies inside one pause (``SILENCE_TOLERANCE`` at each edge)."""
        i = bisect.bisect_right(self.starts, start + SILENCE_TOLERANCE) - 1
        return i >= 0 and end <= self.spans[i][1] + SILENCE_TOLERANCE


def _no_speech(seg: _Seg | None) -> bool:
    """The provider itself judged ``seg`` likely not speech (unknown → speech)."""
    p = seg.no_speech_prob if seg is not None else None
    return p is not None and _finite(p) and p >= SILENCE_NO_SPEECH_PROB


def _owns(plan: _PlanLike, mid: float) -> bool:
    return plan.keep_start <= mid < plan.keep_end


def _local_segments(plan: _PlanLike, res: RawChunkResult, pos: int, fixes: Counter[str],
                    silences: _Silences) -> list[_Seg]:
    segs: list[_Seg] = []
    for k, s in enumerate(res.segments):
        start, end = plan.start + float(s.start), plan.start + float(s.end)
        if not _finite(start, end):
            fixes["invalid_segments_dropped"] += 1
            continue
        if end < start:
            end = start
            fixes["inverted_segments"] += 1
        segs.append(_Seg(chunk_index=plan.index, chunk_pos=pos, local_idx=k, text=(s.text or "").strip(),
                         start=start, end=end, keep_start=plan.keep_start, keep_end=plan.keep_end,
                         no_speech_prob=s.no_speech_prob, avg_logprob=s.avg_logprob,
                         mid_kept=_owns(plan, (start + end) / 2)))
        segs[-1].silent = _no_speech(segs[-1]) and silences.contains(start, end)
    segs.sort(key=lambda s: (s.start, s.end, s.local_idx))
    return segs


def _attach(mid: float, segs: list[_Seg], starts: list[float]) -> _Seg | None:
    """Segment containing ``mid`` (half-open containment preferred), else the nearest one."""
    if not segs:
        return None

    def key(i: int) -> tuple[int, float, int]:
        s = segs[i]
        if s.start <= mid <= s.end:
            return (0, 0.0 if (mid < s.end or s.start == s.end) else 1.0, i)
        return (1, s.start - mid if mid < s.start else mid - s.end, i)

    pos = bisect.bisect_right(starts, mid)
    near = range(max(0, pos - 4), min(len(segs), pos + 2))
    best = min(near, key=key, default=None)
    if best is None or key(best)[0] != 0:
        best = min(range(len(segs)), key=key)
    return segs[best]


def _chunk_words(plan: _PlanLike, res: RawChunkResult, pos: int, segs: list[_Seg], fixes: Counter[str],
                 silences: _Silences) -> _Group:
    """Offset, attach, align and filter one chunk's words (``kept`` = the words this chunk owns).

    A word the provider placed wholly inside a detected pause, in a segment it judged to be no
    speech, is a hallucination: it is left out of every list here (so seam matching never sees
    it) and its segment, if no other word remains in it, is not published.
    """
    items: list[tuple[str, float, float]] = []
    for w in res.words:
        start, end = plan.start + float(w.start), plan.start + float(w.end)
        if not _finite(start, end):
            fixes["invalid_words_dropped"] += 1
            continue
        items.append(((w.text or "").strip(), start, end))
    order = sorted(range(len(items)), key=lambda i: (items[i][1], items[i][2], i))
    starts = [s.start for s in segs]
    owner: dict[int, _Seg | None] = {}
    by_seg: dict[int, list[int]] = {}
    for i in order:
        seg = _attach((items[i][1] + items[i][2]) / 2, segs, starts)
        owner[i] = seg
        if seg is not None:
            seg.n_local_words += 1
            by_seg.setdefault(id(seg), []).append(i)
    display = {i: items[i][0] for i in order}
    for seg in segs:
        members = by_seg.get(id(seg), [])
        if members:
            aligned = align_punctuation([items[i][0] for i in members], seg.text)
            display.update(zip(members, aligned))

    group = _Group(plan=plan, pos=pos, n_segments=len(res.segments), segs=segs)
    kept_idx = [i for i in order if _owns(plan, (items[i][1] + items[i][2]) / 2)]
    moved = sum(1 for x, y in zip(kept_idx, sorted(kept_idx)) if x != y)
    if moved:  # provider returned this chunk's words out of time order
        fixes["reordered_words"] += moved
    for i in order:
        raw, start, end = items[i]
        mid = (start + end) / 2
        owned = _owns(plan, mid)
        if not raw:
            if owned:
                fixes["empty_words_dropped"] += 1
            continue
        if _no_speech(owner[i]) and silences.contains(start, end):
            if owned:
                fixes["dropped_in_silence"] += 1
            continue
        w = _W(raw=raw, text=display[i].strip() or raw, start=start, end=end,
               chunk_index=plan.index, chunk_pos=pos, seg=owner[i])
        group.heard.append(w)
        if owned:
            group.kept.append(w)
        elif plan.keep_start - SEAM_MATCH_SECONDS <= mid < plan.keep_start:
            group.early.append(w)
        elif plan.keep_end <= mid < plan.keep_end + SEAM_MATCH_SECONDS:
            group.late.append(w)
    return group


def _add_synthetic_segments(group: _Group, fixes: Counter[str]) -> None:
    """Give kept words of a chunk that returned words but no segments a synthetic segment."""
    orphans = [w for w in group.kept if w.seg is None]
    if orphans:
        plan = group.plan
        synthetic = _Seg(chunk_index=plan.index, chunk_pos=group.pos, local_idx=group.n_segments,
                         text=" ".join(w.text for w in orphans), start=orphans[0].start,
                         end=max(w.end for w in orphans), keep_start=plan.keep_start,
                         keep_end=plan.keep_end, synthetic=True, n_local_words=len(orphans))
        for w in orphans:
            w.seg = synthetic
        group.segs.append(synthetic)
        fixes["synthetic_segments"] += 1


def _clamp_and_sort(words: list[_W], duration: float, fixes: Counter[str]) -> list[_W]:
    for w in words:
        start, end = min(max(w.start, 0.0), duration), min(max(w.end, 0.0), duration)
        if (start, end) != (w.start, w.end):
            fixes["clamped_times"] += 1
        if end < start:
            end = start
            fixes["inverted_times"] += 1
        w.start, w.end = start, end
    ordered = sorted(words, key=lambda w: (w.start, w.end))
    moved = sum(1 for x, y in zip(words, ordered) if x is not y)
    if moved:
        fixes["reordered_words"] += moved
    return ordered


def _wmid(w: _W) -> float:
    return (w.start + w.end) / 2


def _same_spoken_word(a: _W, b: _W) -> bool:
    if normalize_token(a.raw) != normalize_token(b.raw):
        return False
    overlap = min(a.end, b.end) - max(a.start, b.start)
    return overlap > 0 or abs(_wmid(a) - _wmid(b)) <= DUPLICATE_MID_TOLERANCE


def _same_occurrence(a: _W, b: _W, prev: _Group, nxt: _Group) -> bool | None:
    """Whether same-token copies ``a`` (from ``prev``) and ``b`` (from ``nxt``) are one occurrence.

    Both chunks transcribed the audio around their seam, so when each heard the word equally
    often there, the k-th copy of one chunk is the k-th copy of the other: "the end of the day"
    pairs the 2nd "the" of each chunk, while "no, no" pairs the 1st of one with the 2nd of the
    other. Two decodes time a word differently, so this order is more reliable than comparing
    times across chunks when the word recurs nearby. None when it cannot decide: copies more
    than ``SEAM_MATCH_SECONDS`` apart or outside the shared audio, different counts, or a
    pairing whose copies are timed more than ``SEAM_ORDER_TOLERANCE`` apart (a copy near the
    edge of the compared stretch was counted by one chunk only).
    """
    if abs(_wmid(a) - _wmid(b)) > SEAM_MATCH_SECONDS:
        return None
    lo, hi = min(a.start, b.start), max(a.end, b.end)
    if lo < nxt.plan.start or hi > prev.plan.end:
        return None
    key = normalize_token(a.raw)
    w_lo = max(nxt.plan.start, lo - SEAM_MATCH_SECONDS)      # audio both chunks transcribed
    w_hi = min(prev.plan.end, hi + SEAM_MATCH_SECONDS)

    def copies(group: _Group) -> list[_W]:
        return sorted((w for w in group.heard if normalize_token(w.raw) == key and w_lo <= _wmid(w) <= w_hi),
                      key=lambda w: (w.start, w.end))

    p, n = copies(prev), copies(nxt)
    ia = next((i for i, w in enumerate(p) if w is a), None)
    ib = next((i for i, w in enumerate(n) if w is b), None)
    if len(p) != len(n) or ia is None or ib is None:
        return None
    if any(abs(_wmid(x) - _wmid(y)) > SEAM_ORDER_TOLERANCE for x, y in zip(p, n)):
        return None
    return ia == ib


def _heard_once(a: _W, b: _W, prev: _Group, nxt: _Group) -> bool:
    """True when ``a`` (from ``prev``) and ``b`` (from ``nxt``) are two timings of one spoken word.

    Used for same-token copies that only touch or lie apart (short or zero-length words). Both
    copies must lie in the audio both chunks transcribed, and each chunk must have heard that
    word only once around them: a genuine repetition ("no, no") is heard twice by both chunks.
    """
    if abs(_wmid(a) - _wmid(b)) > SEAM_MATCH_SECONDS:
        return False
    lo, hi = min(a.start, b.start), max(a.end, b.end)
    if lo < nxt.plan.start or hi > prev.plan.end:
        return False
    key = normalize_token(a.raw)

    def copies(group: _Group) -> int:
        return sum(1 for w in group.heard if normalize_token(w.raw) == key
                   and lo - SEAM_HEARD_MARGIN <= _wmid(w) <= hi + SEAM_HEARD_MARGIN)

    return copies(prev) == 1 and copies(nxt) == 1


def _seam_copy(a: _W, b: _W, prev: _Group, nxt: _Group) -> bool:
    """True when ``a`` (from ``prev``) and ``b`` (from ``nxt``) are two chunks' timings of one word."""
    if normalize_token(a.raw) != normalize_token(b.raw):
        return False
    same = _same_occurrence(a, b, prev, nxt)
    if same is not None:
        return same
    return _same_spoken_word(a, b) or _heard_once(a, b, prev, nxt)


def _seam_duplicates(prev: _Group, nxt: _Group) -> int:
    """Number of leading words of ``nxt`` that repeat the trailing words of ``prev`` at a seam."""
    best = 0
    tail, head = prev.kept, nxt.kept
    for k in range(1, min(len(tail), len(head), DUPLICATE_MAX_WORDS) + 1):
        if all(_seam_copy(a, b, prev, nxt) for a, b in zip(tail[-k:], head[:k])):
            best = k
    return best


def _drop_seam_duplicates(groups: list[_Group], fixes: Counter[str]) -> None:
    prev: _Group | None = None
    for group in groups:
        if prev is not None and group.kept:
            k = _seam_duplicates(prev, group)
            if k:
                del group.kept[:k]
                fixes["duplicate_words_dropped"] += k
        if group.kept:
            prev = group


def _recover_seam_words(groups: list[_Group], fixes: Counter[str]) -> None:
    """Keep a seam word that each neighbouring chunk timed into the other's keep range.

    Chunk A times the word with its midpoint at/after the cut and chunk B before it, so
    neither owns it. When A's copy is the same occurrence as one of B's (by occurrence order,
    which also proves neither chunk kept it), or else matches one of B's and no kept word of
    either chunk near the seam is that same spoken word (timed over the pair, or a seam copy
    of either), A's copy (its own provider timing) is kept. Another occurrence of a common
    word nearby ("the cat and the dog") does not block the recovery.
    """
    for prev, nxt in zip(groups, groups[1:]):
        used: set[int] = set()
        near = prev.kept[-DUPLICATE_MAX_WORDS:] + nxt.kept[:DUPLICATE_MAX_WORDS]
        for a in prev.late:
            match = next((b for b in nxt.early if id(b) not in used and _seam_copy(a, b, prev, nxt)), None)
            if match is None:
                continue
            key = normalize_token(a.raw)
            lo, hi = min(a.start, match.start), max(a.end, match.end)
            if _same_occurrence(a, match, prev, nxt) is None and any(
                    normalize_token(k.raw) == key and (min(k.end, hi) - max(k.start, lo) >= 0
                                                       or _same_spoken_word(k, a) or _same_spoken_word(k, match))
                    for k in near):
                continue                                  # a chunk kept this word after all
            used.add(id(match))
            bisect.insort(prev.kept, a, key=lambda w: (w.start, w.end))
            fixes["seam_words_recovered"] += 1


def _enforce_monotonic(words: list[_W], fixes: Counter[str]) -> None:
    for prev, w in zip(words, words[1:]):
        if w.start < prev.end:
            w.start = prev.end
            if w.end < w.start:
                w.end = w.start
            fixes["overlapping_words"] += 1


def _wordless_kept(s: _Seg) -> bool:
    """A provider segment without any word timings, owned by this chunk (kept for completeness)."""
    return s.mid_kept and bool(s.text) and s.n_local_words == 0 and not s.silent


def _ordered_segments(words: list[_W], segs: list[_Seg], fixes: Counter[str]) -> list[_Seg]:
    """Segments to publish (in time order), with each word's segment non-decreasing along the words."""
    used = {id(w.seg) for w in words}
    emitted = [s for s in segs if id(s) in used or _wordless_kept(s)]
    emitted.sort(key=lambda s: (s.chunk_pos, s.start, s.local_idx))
    ordinal = {id(s): i for i, s in enumerate(emitted)}
    current = -1
    for i, w in enumerate(words):
        assert w.seg is not None
        o = ordinal[id(w.seg)]
        if o < current:
            w.seg.modified = True
            w.seg = emitted[current]
            w.seg.modified = True
            fixes["words_reassigned_segment"] += 1
        else:
            current = o
        w.seg.members.append(i)
    merged: list[_Seg] = []
    for s in emitted:
        if not (s.members or _wordless_kept(s)):
            continue
        if merged and _seam_copies(merged[-1], s):
            _absorb(merged[-1], s, words)
            continue
        merged.append(s)
    return merged


def _seam_copies(a: _Seg, b: _Seg) -> bool:
    """True when ``a`` and ``b`` are two chunks' copies of one provider segment straddling a seam."""
    if b.chunk_pos != a.chunk_pos + 1 or not a.members or not b.members or a.members[-1] + 1 != b.members[0]:
        return False
    shorter = min(a.end - a.start, b.end - b.start)
    overlap = min(a.end, b.end) - max(a.start, b.start)
    return shorter > 0 and overlap >= 0.5 * shorter


def _absorb(a: _Seg, b: _Seg, words: list[_W]) -> None:
    """Merge seam copy ``b`` (whose words directly follow ``a``'s) into ``a``."""
    for i in b.members:
        words[i].seg = a
    a.members.extend(b.members)
    a.modified = True
    a.start, a.end = min(a.start, b.start), max(a.end, b.end)
    a.keep_start, a.keep_end = min(a.keep_start, b.keep_start), max(a.keep_end, b.keep_end)
    if b.mid_kept and not a.mid_kept:   # report the chunk that owns the segment's midpoint
        a.chunk_index, a.no_speech_prob, a.avg_logprob = b.chunk_index, b.no_speech_prob, b.avg_logprob
        a.mid_kept = True


def _segment_models(words: list[_W], segs: list[_Seg], duration: float,
                    fixes: Counter[str]) -> list[Segment]:
    out: list[Segment] = []
    prev_start = 0.0
    for n, s in enumerate(segs, start=1):
        start = min(max(max(s.start, s.keep_start), 0.0), duration)
        end = min(max(min(s.end, s.keep_end), 0.0), duration)
        if s.members:
            start = min(start, words[s.members[0]].start)
            end = max(end, words[s.members[-1]].end)
        end = max(end, start)
        if start < prev_start:
            start = prev_start
            end = max(end, start)
            fixes["segment_times_adjusted"] += 1
        prev_start = start
        complete = not s.synthetic and not s.modified and len(s.members) == s.n_local_words
        text = s.text if (complete or not s.members) else ""
        if not text:
            text = " ".join(words[i].text for i in s.members)
        out.append(Segment(id=segment_id(n), start=start, end=end, text=text,
                           word_start=s.members[0] if s.members else -1,
                           word_end=s.members[-1] if s.members else -1,
                           chunk_index=s.chunk_index, no_speech_prob=s.no_speech_prob,
                           avg_logprob=s.avg_logprob))
    return out


def merge_chunks(results: list[tuple["ChunkPlan", RawChunkResult]], *, duration: float, provider: str,
                 model: str, language: str, chunk_meta: list[TranscriptChunk],
                 silences: Sequence[tuple[float, float]] | None = None) -> Transcript:
    """Merge per-chunk provider results into one source-time :class:`Transcript`.

    Steps: offset by ``plan.start``; attach words to their chunk-local segment by time and
    align punctuation from that segment's text; keep words/segments whose midpoint lies in
    the chunk's keep range; drop empty words and silence hallucinations (a word wholly inside
    one of ``silences`` — source-second pauses from ``silencedetect`` at -35 dB / 0.35 s —
    whose segment has ``no_speech_prob >= 0.5``; a segment left without words goes too);
    recover a seam word neither neighbouring chunk kept; clamp to ``[0, duration]``; sort
    (chunks in keep order, words by start within a chunk); drop seam duplicates (a run of
    identical words at the same time in two neighbouring chunks); make words monotonic and
    non-overlapping; assign IDs; build units.
    Fix counts land in ``Transcript.fixes``.
    """
    if not _finite(duration) or duration < 0:
        raise ValueError("duration must be a finite, non-negative number of seconds")
    fixes: Counter[str] = Counter()
    pauses = _Silences(silences)
    ordered = sorted(results, key=lambda r: (r[0].keep_start, r[0].index))
    groups = [_chunk_words(plan, res, pos, _local_segments(plan, res, pos, fixes, pauses), fixes, pauses)
              for pos, (plan, res) in enumerate(ordered)]
    _recover_seam_words(groups, fixes)
    all_segs: list[_Seg] = []
    for g in groups:
        _add_synthetic_segments(g, fixes)
        all_segs.extend(g.segs)
        fixes["segments_dropped_in_silence"] += sum(
            1 for s in g.segs if s.silent and s.mid_kept and s.text and s.n_local_words == 0)
        g.kept = _clamp_and_sort(g.kept, duration, fixes)
    _drop_seam_duplicates(groups, fixes)
    flat = [w for g in groups for w in g.kept]
    _enforce_monotonic(flat, fixes)
    segs_out = _ordered_segments(flat, all_segs, fixes)
    segments = _segment_models(flat, segs_out, duration, fixes)
    seg_ids = {id(s): segments[k].id for k, s in enumerate(segs_out)}
    ids = word_ids_for([w.start for w in flat])
    words = [Word(id=wid, text=w.text, raw=w.raw, start=w.start, end=w.end,
                  segment_id=seg_ids[id(w.seg)], chunk_index=w.chunk_index)
             for wid, w in zip(ids, flat)]
    return Transcript(version=TRANSCRIPT_VERSION, provider=provider, model=model, language=language,
                      duration=float(duration), words=words, segments=segments, units=build_units(words),
                      chunks=sorted(chunk_meta, key=lambda c: c.index),
                      fixes={k: v for k, v in sorted(fixes.items()) if v}, created_at=time.time())


# ----------------------------------------------------------------------------- validation
def _check_ids(kind: str, ids: list[str], fmt: Any) -> None:
    for n, value in enumerate(ids, start=1):
        if value != fmt(n):
            raise ValueError(f"{kind} #{n} has id {value!r}, expected {fmt(n)!r}")


def _check_partition(kind: str, ranges: list[tuple[int, int]], n_words: int) -> None:
    """``ranges`` (in order) must tile ``0..n_words-1`` exactly."""
    expected = 0
    for k, (a, b) in enumerate(ranges):
        if a != expected or b < a or b >= n_words:
            raise ValueError(f"{kind} #{k + 1} word range [{a}, {b}] does not continue at word {expected}")
        expected = b + 1
    if expected != n_words:
        raise ValueError(f"{kind} word ranges cover {expected} of {n_words} words")


def validate_transcript(t: Transcript) -> None:
    """Raise ``ValueError`` if times are non-monotonic/out of bounds or IDs/ranges are inconsistent."""
    if not _finite(t.duration) or t.duration < 0:
        raise ValueError("transcript duration is invalid")
    hi = t.duration + _EPS
    if [w.id for w in t.words] != word_ids_for([w.start for w in t.words]):
        raise ValueError("word ids do not match their start times")
    _check_ids("segment", [s.id for s in t.segments], segment_id)
    _check_ids("unit", [u.id for u in t.units], unit_id)
    prev_end = 0.0
    for w in t.words:
        if not w.text.strip():
            raise ValueError(f"word {w.id} has empty text")
        if not _finite(w.start, w.end) or w.start < -_EPS or w.end > hi or w.end < w.start:
            raise ValueError(f"word {w.id} has invalid times [{w.start}, {w.end}]")
        if w.start < prev_end - _EPS:
            raise ValueError(f"word {w.id} starts at {w.start} before the previous word ends ({prev_end})")
        prev_end = w.end
    prev_start = 0.0
    seg_ranges: list[tuple[int, int]] = []
    for s in t.segments:
        if not _finite(s.start, s.end) or s.start < -_EPS or s.end > hi or s.end < s.start:
            raise ValueError(f"segment {s.id} has invalid times [{s.start}, {s.end}]")
        if s.start < prev_start - _EPS:
            raise ValueError(f"segment {s.id} is out of time order")
        prev_start = s.start
        if (s.word_start < 0) != (s.word_end < 0):
            raise ValueError(f"segment {s.id} has a half-empty word range")
        if s.word_start >= 0:
            seg_ranges.append((s.word_start, s.word_end))
            for w in t.words[s.word_start:s.word_end + 1]:
                if w.segment_id != s.id:
                    raise ValueError(f"word {w.id} lies in segment {s.id} but names {w.segment_id}")
    _check_partition("segment", seg_ranges, len(t.words))
    _check_partition("unit", [(u.word_start, u.word_end) for u in t.units], len(t.words))
    for u in t.units:
        first, last = t.words[u.word_start], t.words[u.word_end]
        if abs(u.start - first.start) > _EPS or abs(u.end - last.end) > _EPS:
            raise ValueError(f"unit {u.id} times do not match its words")
        if not u.text.strip():
            raise ValueError(f"unit {u.id} has empty text")
    indices = [c.index for c in t.chunks]
    if len(set(indices)) != len(indices):
        raise ValueError("transcript chunks have duplicate indices")


# ----------------------------------------------------------------------------- lookups
def _mid(w: Word) -> float:
    return (w.start + w.end) / 2


def words_between(t: Transcript, src_start: float, src_end: float) -> list[Word]:
    """Words whose midpoint lies in ``[src_start, src_end)`` (source seconds)."""
    if src_end <= src_start:
        return []
    lo = bisect.bisect_left(t.words, src_start, key=_mid)
    hi = bisect.bisect_left(t.words, src_end, key=_mid)
    return list(t.words[lo:hi])


def unit_index(t: Transcript) -> dict[str, int]:
    """Map unit ID → position in ``t.units``."""
    return {u.id: i for i, u in enumerate(t.units)}


def range_for_units(t: Transcript, start_unit_id: str, end_unit_id: str, *,
                    pad: float = 0.25) -> tuple[float, float]:
    """Source range ``(src_start, src_end)`` covering the units, padded into the surrounding pauses.

    ``src_start = first_word.start - min(pad, gap_to_previous_word / 2)`` and likewise at
    the end (full ``pad`` at the very start/end of the transcript), clamped to
    ``[0, duration]``. Raises ``ValueError`` for unknown IDs or a reversed range.

    Clips select words by midpoint in the half-open ``[src_start, src_end)``
    (:func:`words_between`, the caption builder). When there is no gap to pad into and a
    boundary word has zero length, its midpoint sits exactly on the boundary, so the edge
    is moved by ``BOUNDARY_EDGE`` (1 ms) to keep the unit's own words inside the range and
    the neighbouring words outside it.
    """
    idx = unit_index(t)
    if start_unit_id not in idx or end_unit_id not in idx:
        missing = start_unit_id if start_unit_id not in idx else end_unit_id
        raise ValueError(f"unknown unit id {missing!r}")
    a, b = idx[start_unit_id], idx[end_unit_id]
    if b < a:
        raise ValueError(f"unit range {start_unit_id}..{end_unit_id} is reversed")
    pad = max(0.0, pad)
    ws, we = t.units[a].word_start, t.units[b].word_end
    first, last = t.words[ws], t.words[we]
    before = pad if ws == 0 else min(pad, max(0.0, first.start - t.words[ws - 1].end) / 2)
    after = pad if we == len(t.words) - 1 else min(pad, max(0.0, t.words[we + 1].start - last.end) / 2)
    src_start = first.start - before
    src_end = max(last.end + after, _mid(last) + BOUNDARY_EDGE)      # zero-length last word stays inside
    if ws > 0 and _mid(t.words[ws - 1]) >= src_start:               # zero-length previous word on the edge
        src_start = min(_mid(t.words[ws - 1]) + BOUNDARY_EDGE, _mid(first))
    src_start = min(max(src_start, 0.0), t.duration)
    src_end = min(max(src_end, 0.0), t.duration)
    return src_start, src_end


# ----------------------------------------------------------------------------- speech amount
def _speech_seconds(words: Sequence[Word]) -> float:
    """Total speaking time: word intervals joined across pauses ≤ ``SPEECH_GAP_BRIDGE``."""
    total = 0.0
    run_start = run_end = None
    for w in words:
        if run_end is not None and w.start - run_end <= SPEECH_GAP_BRIDGE:
            run_end = max(run_end, w.end)
            continue
        if run_start is not None and run_end is not None:
            total += run_end - run_start
        run_start, run_end = w.start, w.end
    if run_start is not None and run_end is not None:
        total += run_end - run_start
    return total


def _stats(words: Sequence[Word], duration: float) -> dict[str, float | int]:
    speech = _speech_seconds(words)
    minutes = duration / 60.0
    return {
        "word_count": len(words),
        "duration": round(duration, 3),
        "speech_seconds": round(speech, 3),
        "coverage": round(min(1.0, speech / duration), 4) if duration > 0 else 0.0,
        "words_per_minute": round(len(words) / minutes, 1) if minutes > 0 else 0.0,
    }


def speech_stats(t: Transcript) -> dict[str, Any]:
    """``word_count``, ``speech_seconds``, ``coverage`` (0–1 of ``t.duration``), ``words_per_minute``."""
    return _stats(t.words, t.duration)


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {sec:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def insufficient_speech_reason(t: Transcript, media_duration: float) -> str | None:
    """A friendly explanation when there is too little speech to pick clips from, else ``None``."""
    duration = media_duration if _finite(media_duration) and media_duration > 0 else t.duration
    stats = _stats(t.words, duration)
    n = int(stats["word_count"])
    if n == 0:
        return "No speech was found in this video, so there is nothing to build clips from. " + TARGET_NOTE
    if n < MIN_WORDS:
        noun = "word of speech was" if n == 1 else "words of speech were"
        return f"Only {n} {noun} found, which is not enough to build clips from. " + TARGET_NOTE
    coverage = float(stats["coverage"])
    if coverage < MIN_COVERAGE:
        pct = "less than 1%" if coverage < 0.01 else f"about {round(coverage * 100)}%"
        return (f"Speech covers only {pct} of this video ({_fmt_duration(float(stats['speech_seconds']))} "
                f"of {_fmt_duration(duration)}), which is too little to build clips from. " + TARGET_NOTE)
    wpm = float(stats["words_per_minute"])
    if wpm < MIN_WORDS_PER_MINUTE:
        return (f"This video has very little speech (about {wpm:.0f} words per minute), which is too "
                "little to build clips from. " + TARGET_NOTE)
    return None
