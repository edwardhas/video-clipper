"""Caption cues for a clip: grouping transcript words, owner text edits, SRT export.

Times: transcript words carry *source* seconds (``start``/``end``); cues and their
words carry *clip-relative* seconds (``rel_*`` = src - edit.src_start).

Rules that matter for honesty of the output:

* Per-word timings always come from the transcript. When the owner rewrites a cue
  with a different number of words, the cue is rendered as one block without a word
  highlight (``highlight=False``) — timings for the new tokens are never invented.
* Upper-casing is a render-time style option and is not applied here.

Cue grouping obeys the character budget (``max_chars`` x ``max_lines``) *and* the
measured width of the text at the clip's caption size, so a normal cue fits on
screen without the renderer having to shrink the font.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..errors import InputError
from ..models import CaptionWord, ClipEdit, Cue, CueOverride, Word
from .caption_image import lines_needed

SENTENCE_END = (".", "!", "?", "…")
_TRAILING_QUOTES = "\"'»”’)]"
# Abbreviations whose trailing period does not end a sentence.
_ABBREVIATIONS = frozenset({"mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "e.g.", "i.e.", "prof.", "jr.", "sr."})

JOIN_GAP = 0.25        # gaps shorter than this between cues are closed (no flicker)
HOLD_AFTER = 0.4       # a cue followed by silence stays up to this much longer
MAX_CUE_TEXT = 300     # CueOverride.text limit


@dataclass
class _Item:
    """A visible clip word: transcript word + display text + clip-relative times."""
    word: Word
    text: str
    rel_start: float
    rel_end: float
    overridden: bool


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _t(v: float) -> float:
    """Round a clip-relative time to whole microseconds (removes float noise like 2.3000000000000007)."""
    return round(v, 6)


def _norm_text(text: str) -> str:
    return " ".join(text.split())


def ends_sentence(text: str) -> bool:
    t = text.rstrip(_TRAILING_QUOTES)
    if not t.endswith(SENTENCE_END):
        return False
    return t.lower() not in _ABBREVIATIONS


def _line_count(texts: list[str], max_chars: int) -> int:
    """Greedy character wrap: number of lines needed for ``texts`` at ``max_chars``."""
    lines = 0
    cur = 0
    for t in texts:
        n = len(t)
        if cur == 0:
            lines += 1
            cur = n
        elif cur + 1 + n <= max_chars:
            cur += 1 + n
        else:
            lines += 1
            cur = n
    return lines


def _clip_words(words: list[Word], edit: ClipEdit) -> list[Word]:
    """Words whose midpoint lies in ``[edit.src_start, edit.src_end)``, in time order."""
    inside = [w for w in words if edit.src_start <= (w.start + w.end) / 2.0 < edit.src_end]
    return sorted(inside, key=lambda w: (w.start, w.end))


def _override_ranges(clip_words: list[Word], overrides: list[CueOverride]) -> dict[int, tuple[int, CueOverride]]:
    """Map start index -> (end index, override) for overrides fully inside the clip.

    Later overrides win when ranges overlap (the most recent edit is appended last).
    """
    index = {w.id: i for i, w in enumerate(clip_words)}
    taken: set[int] = set()
    out: dict[int, tuple[int, CueOverride]] = {}
    for ov in reversed(overrides):
        i, j = index.get(ov.first_word_id), index.get(ov.last_word_id)
        if i is None or j is None or j < i:
            continue
        span = set(range(i, j + 1))
        if span & taken:
            continue
        taken |= span
        out[i] = (j, ov)
    return out


def _cue_from_items(items: list[_Item]) -> Cue:
    first, last = items[0], items[-1]
    return Cue(
        id=first.word.id,
        first_word_id=first.word.id,
        last_word_id=last.word.id,
        text=" ".join(it.text for it in items),
        words=[CaptionWord(word_id=it.word.id, text=it.text, rel_start=it.rel_start, rel_end=it.rel_end)
               for it in items],
        rel_start=first.rel_start,
        rel_end=last.rel_end,
        highlight=True,
        edited=any(it.overridden for it in items),
    )


@dataclass(frozen=True)
class _Rules:
    max_chars: int
    max_lines: int
    max_cue_seconds: float
    pause_break: float
    measured_lines: Callable[[list[str]], int] | None = None    # lines at the rendered font size


def _should_break(group: list[_Item], nxt: _Item, rules: _Rules) -> bool:
    prev = group[-1]
    if ends_sentence(prev.text):
        return True
    if nxt.word.start - prev.word.end >= rules.pause_break:
        return True
    texts = [it.text for it in group] + [nxt.text]
    if _line_count(texts, rules.max_chars) > rules.max_lines:
        return True
    if rules.measured_lines is not None and rules.measured_lines(texts) > rules.max_lines:
        return True
    if nxt.word.end - group[0].word.start > rules.max_cue_seconds:
        return True
    return False


def _group(items: list[_Item], rules: _Rules) -> list[Cue]:
    cues: list[Cue] = []
    group: list[_Item] = []
    for it in items:
        if group and _should_break(group, it, rules):
            cues.append(_cue_from_items(group))
            group = []
        group.append(it)
    if group:
        cues.append(_cue_from_items(group))
    return cues


def _adjust_timing(cues: list[Cue], duration: float) -> list[Cue]:
    """Close tiny gaps, hold cues briefly into silence, never overlap or pass the clip end."""
    cues = sorted(cues, key=lambda c: (c.rel_start, c.rel_end))
    out: list[Cue] = []
    for i, cue in enumerate(cues):
        end = cue.rel_end
        if i + 1 < len(cues):
            nxt_start = cues[i + 1].rel_start
            gap = nxt_start - end
            if gap < JOIN_GAP:
                end = nxt_start
            else:
                end = min(end + HOLD_AFTER, nxt_start)
        else:
            end = end + HOLD_AFTER
        end = _t(_clamp(end, cue.rel_start, duration))
        out.append(cue.model_copy(update={"rel_end": end}))
    return out


def _hidden_cues(clip_words: list[Word], overrides: dict[int, tuple[int, CueOverride]], edit: ClipEdit,
                 rules: _Rules, rel: Callable[[float], float]) -> list[Cue]:
    """Empty-text, ``edited`` cues for the words the owner hid (editor display only).

    Runs of consecutive hidden words are grouped with the same rules as visible cues
    (using the transcript text), so a hidden caption comes back roughly as it was shown.
    """
    covered = {k for i, (j, _) in overrides.items() for k in range(i, j + 1)}
    hidden: list[Cue] = []
    run: list[_Item] = []

    def flush() -> None:
        for cue in _group(run, rules):
            hidden.append(cue.model_copy(update={"text": "", "words": [], "highlight": False, "edited": True}))
        run.clear()

    for i, w in enumerate(clip_words):
        if i in covered:
            flush()
            if i in overrides:
                j, ov = overrides[i]
                if not _norm_text(ov.text):
                    first, last = clip_words[i], clip_words[j]
                    start = rel(first.start)
                    hidden.append(Cue(id=first.id, first_word_id=first.id, last_word_id=last.id, text="",
                                      words=[], rel_start=start, rel_end=max(start, rel(last.end)),
                                      highlight=False, edited=True))
            continue
        override = edit.word_overrides.get(w.id)
        if override is not None and not _norm_text(override):
            start = rel(w.start)
            run.append(_Item(word=w, text=_norm_text(w.text) or "-", rel_start=start,
                             rel_end=max(start, rel(w.end)), overridden=True))
        else:
            flush()
    flush()
    return hidden


def build_cues(words: list[Word], edit: ClipEdit, *, max_chars: int = 32, max_lines: int = 2,
               max_cue_seconds: float = 3.0, pause_break: float = 0.6, include_hidden: bool = False) -> list[Cue]:
    """Group the clip's transcript words into caption cues (clip-relative times).

    ``include_hidden`` (editor display only, never for rendering or SRT) also returns an
    empty-text, ``edited`` cue for each caption the owner hid, so it can be shown and restored.
    The visible cues are identical either way.
    """
    duration = edit.src_end - edit.src_start
    clip_words = _clip_words(words, edit)
    overrides = _override_ranges(clip_words, edit.cue_overrides)
    style = edit.style
    rules = _Rules(max_chars=max_chars, max_lines=max_lines, max_cue_seconds=max_cue_seconds,
                   pause_break=pause_break, measured_lines=lambda texts: lines_needed(texts, style))

    def rel(src_t: float) -> float:
        return _t(_clamp(src_t - edit.src_start, 0.0, duration))

    cues: list[Cue] = []
    pending: list[_Item] = []
    i = 0
    while i < len(clip_words):
        if i in overrides:
            j, ov = overrides[i]
            cues.extend(_group(pending, rules))
            pending = []
            text = _norm_text(ov.text)
            if text:
                first, last = clip_words[i], clip_words[j]
                start = rel(first.start)
                cues.append(Cue(id=first.id, first_word_id=first.id, last_word_id=last.id, text=text, words=[],
                                rel_start=start, rel_end=max(start, rel(last.end)), highlight=False, edited=True))
            i = j + 1
            continue
        w = clip_words[i]
        override = edit.word_overrides.get(w.id)
        text = _norm_text(override if override is not None else w.text)
        if text:
            start = rel(w.start)
            pending.append(_Item(word=w, text=text, rel_start=start, rel_end=max(start, rel(w.end)),
                                 overridden=override is not None and override != w.text))
        i += 1
    cues.extend(_group(pending, rules))
    cues = _adjust_timing(cues, duration)
    if include_hidden:
        cues = sorted(cues + _hidden_cues(clip_words, overrides, edit, rules, rel),
                      key=lambda c: (c.rel_start, c.rel_end))
    return cues


# ----------------------------------------------------------------------------- editing
def _range_indices(words: list[Word], first_word_id: str, last_word_id: str) -> tuple[int, int]:
    index = {w.id: i for i, w in enumerate(words)}
    i, j = index.get(first_word_id), index.get(last_word_id)
    if i is None or j is None or j < i:
        raise InputError("That caption no longer matches the transcript. Reload the clip and try again.")
    return i, j


def _overlaps(ov: CueOverride, index: dict[str, int], i: int, j: int) -> bool:
    a, b = index.get(ov.first_word_id), index.get(ov.last_word_id)
    if a is None or b is None:
        return False
    return a <= j and b >= i


def apply_caption_edit(edit: ClipEdit, words: list[Word], first_word_id: str, last_word_id: str,
                       text: str) -> ClipEdit:
    """Return a new edit with the owner's text for words ``first..last`` (inclusive).

    * same token count as the spoken words -> per-word overrides (timing + highlight kept)
    * different token count -> one ``CueOverride`` (rendered without word highlight)
    * empty text -> the words are hidden (override ``""`` for each)
    Any cue override overlapping the range is replaced.
    """
    words = sorted(words, key=lambda w: (w.start, w.end))
    i, j = _range_indices(words, first_word_id, last_word_id)
    span = words[i:j + 1]
    index = {w.id: k for k, w in enumerate(words)}
    tokens = text.split()

    word_overrides = dict(edit.word_overrides)
    cue_overrides = [ov for ov in edit.cue_overrides if not _overlaps(ov, index, i, j)]

    if not tokens:
        for w in span:
            word_overrides[w.id] = ""
    elif len(tokens) == len(span):
        for w, tok in zip(span, tokens):
            if tok == w.text:
                word_overrides.pop(w.id, None)
            else:
                word_overrides[w.id] = tok
    else:
        joined = " ".join(tokens)
        if len(joined) > MAX_CUE_TEXT:
            raise InputError(f"Caption text can be at most {MAX_CUE_TEXT} characters.")
        for w in span:
            word_overrides.pop(w.id, None)
        cue_overrides.append(CueOverride(first_word_id=span[0].id, last_word_id=span[-1].id, text=joined))
    data = edit.model_dump()
    data["word_overrides"] = word_overrides
    data["cue_overrides"] = [ov.model_dump() for ov in cue_overrides]
    return ClipEdit.model_validate(data)


# ----------------------------------------------------------------------------- SRT
def _srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    """Standard SRT (1-based, ``HH:MM:SS,mmm``, blank-line separated, ``\\n`` endings)."""
    blocks: list[str] = []
    for cue in sorted(cues, key=lambda c: c.rel_start):
        text = _norm_text(cue.text)
        if not text:
            continue
        blocks.append(f"{len(blocks) + 1}\n{_srt_time(cue.rel_start)} --> {_srt_time(cue.rel_end)}\n{text}\n")
    return "\n".join(blocks)
