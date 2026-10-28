"""Caption PNG states and the ffconcat timeline (Pillow only, no FFmpeg)."""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from clipper.models import CaptionStyle, CaptionWord, Cue
from clipper.render.caption_image import (
    BLANK_NAME,
    BOTTOM_SAFE,
    SIDE_MARGIN,
    TOP_SAFE,
    lines_needed,
    render_caption_track,
)

AMBER = np.array([0xF2, 0xA3, 0x3A])


def mk_cue(words: list[tuple[str, float, float]], *, highlight: bool = True, end: float | None = None,
           text: str | None = None, idx: int = 1) -> Cue:
    cws = [CaptionWord(word_id=f"w{idx + i:06d}", text=t, rel_start=s, rel_end=e) for i, (t, s, e) in enumerate(words)]
    return Cue(id=cws[0].word_id if cws else f"w{idx:06d}", first_word_id=cws[0].word_id if cws else f"w{idx:06d}",
               last_word_id=cws[-1].word_id if cws else f"w{idx:06d}",
               text=text if text is not None else " ".join(t for t, _, _ in words),
               words=cws if highlight else [], rel_start=words[0][1], rel_end=end if end is not None else words[-1][2],
               highlight=highlight)


def parse_concat(path: Path) -> list[tuple[str, float | None]]:
    """(file, duration) entries; also checks that every file line is followed by the 1 ms
    ``option framerate 1000`` (without it the concat demuxer rounds switches to 40 ms)."""
    lines = path.read_text().splitlines()
    assert lines[0] == "ffconcat version 1.0"
    entries: list[tuple[str, float | None]] = []
    options: list[str] = []
    for line in lines[1:]:
        if m := re.fullmatch(r"file '([^'/]+)'", line):
            entries.append((m.group(1), None))
            options.append("")
        elif m := re.fullmatch(r"option (.+)", line):
            options[-1] = m.group(1)
        elif m := re.fullmatch(r"duration (\d+\.\d+)", line):
            name, _ = entries[-1]
            entries[-1] = (name, float(m.group(1)))
        else:
            raise AssertionError(f"unexpected ffconcat line: {line!r}")
    assert options and all(o == "framerate 1000" for o in options), options
    return entries


def alpha(path: Path) -> np.ndarray:
    img = Image.open(path)
    assert img.mode == "RGBA"
    return np.asarray(img)[:, :, 3]


@pytest.fixture
def two_cues() -> list[Cue]:
    return [
        mk_cue([("Clear", 0.5, 0.9), ("ideas", 1.0, 1.4), ("win.", 1.5, 1.9)], end=2.2),
        mk_cue([("Every", 3.0, 3.4), ("time.", 3.5, 3.9)], end=4.3, idx=10),
    ]


def test_pngs_share_band_size_and_are_transparent_outside_text(tmp_path: Path, two_cues: list[Cue]) -> None:
    track = render_caption_track(two_cues, CaptionStyle(), tmp_path, duration=5.0)
    assert track is not None
    pngs = sorted(tmp_path.glob("*.png"))
    assert len(pngs) == track.frames == 3 + 2 + 1          # one per highlighted word + blank
    for p in pngs:
        with Image.open(p) as img:
            assert img.size == (track.w, track.h) == (1080, track.h)
            assert img.mode == "RGBA"
    assert (alpha(tmp_path / BLANK_NAME) == 0).all()
    a = alpha(sorted(tmp_path.glob("cap_*.png"))[0])
    assert a.any()
    assert (a[:, :SIDE_MARGIN] == 0).all() and (a[:, -SIDE_MARGIN:] == 0).all()
    assert (a[0, :] == 0).all() and (a[-1, :] == 0).all()
    assert (a == 0).mean() > 0.5                           # mostly transparent band


def test_highlight_states_colour_only_the_spoken_word(tmp_path: Path) -> None:
    cue = mk_cue([("one", 0.0, 0.5), ("two", 0.5, 1.0)])
    track = render_caption_track([cue], CaptionStyle(color="white"), tmp_path, duration=1.0)
    assert track is not None
    entries = parse_concat(track.concat_path)
    first, second = (np.asarray(Image.open(tmp_path / n).convert("RGB")).astype(int) for n, _ in entries[:2])

    def amber_cols(img: np.ndarray) -> np.ndarray:
        mask = (np.abs(img - AMBER).sum(axis=2) < 40)
        return np.nonzero(mask.any(axis=0))[0]

    c1, c2 = amber_cols(first), amber_cols(second)
    assert len(c1) and len(c2)
    assert c1.max() < c2.min()                             # word 1 lit first, then word 2 (to its right)


def test_highlight_disabled_by_style_or_cue(tmp_path: Path) -> None:
    words = [("a", 0.0, 0.3), ("b", 0.3, 0.6), ("c", 0.6, 0.9)]
    t1 = render_caption_track([mk_cue(words)], CaptionStyle(highlight=False), tmp_path / "a", duration=1.0)
    t2 = render_caption_track([mk_cue(words, highlight=False)], CaptionStyle(), tmp_path / "b", duration=1.0)
    assert t1 is not None and t2 is not None
    assert t1.frames == 2 and t2.frames == 2               # one cue state + blank


def test_identical_states_are_written_once(tmp_path: Path) -> None:
    cues = [mk_cue([("same", 0.0, 0.5)], highlight=False), mk_cue([("same", 1.0, 1.5)], highlight=False, idx=5)]
    track = render_caption_track(cues, CaptionStyle(), tmp_path, duration=2.0)
    assert track is not None and track.frames == 2
    names = [n for n, _ in parse_concat(track.concat_path)]
    assert names[0] == names[2] != names[1] == BLANK_NAME


def test_ffconcat_covers_exact_duration_with_transparent_gaps(tmp_path: Path, two_cues: list[Cue]) -> None:
    duration = 5.123457
    track = render_caption_track(two_cues, CaptionStyle(), tmp_path, duration=duration)
    assert track is not None
    entries = parse_concat(track.concat_path)
    timed = entries[:-1]
    assert all(d is not None and d > 0 for _, d in timed)
    assert sum(d for _, d in timed) == pytest.approx(duration, abs=0.001)
    assert entries[0][0] == BLANK_NAME and entries[0][1] == pytest.approx(0.5)       # gap before first cue
    assert entries[-1] == (BLANK_NAME, None)               # trailing repeat (concat quirk) is transparent
    assert entries[-2][0] == BLANK_NAME                    # last timed state is transparent too
    starts = list(itertools.accumulate([0.0] + [d for _, d in timed]))
    gap_index = next(i for i, s in enumerate(starts) if s == pytest.approx(2.2))
    assert timed[gap_index][0] == BLANK_NAME and timed[gap_index][1] == pytest.approx(0.8)
    for name, _ in entries:
        assert (tmp_path / name).is_file()


def test_word_lit_until_next_word_starts(tmp_path: Path) -> None:
    cue = mk_cue([("a", 1.0, 1.2), ("b", 1.6, 1.8)], end=2.0)
    track = render_caption_track([cue], CaptionStyle(), tmp_path, duration=3.0)
    assert track is not None
    durations = [d for _, d in parse_concat(track.concat_path)[:-1]]
    assert durations == pytest.approx([1.0, 0.6, 0.4, 1.0])


def test_cue_clipped_at_track_end(tmp_path: Path) -> None:
    cue = mk_cue([("tail", 1.5, 2.5)], highlight=False)
    track = render_caption_track([cue], CaptionStyle(), tmp_path, duration=2.0)
    assert track is not None
    entries = parse_concat(track.concat_path)
    assert [d for _, d in entries[:-1]] == pytest.approx([1.5, 0.5])
    assert entries[-1] == (BLANK_NAME, None)


def test_nothing_to_show_returns_none(tmp_path: Path) -> None:
    assert render_caption_track([], CaptionStyle(), tmp_path, duration=3.0) is None
    empty = mk_cue([("x", 0.5, 1.0)], highlight=False, text="   ")
    empty = empty.model_copy(update={"words": []})
    assert render_caption_track([empty], CaptionStyle(), tmp_path, duration=3.0) is None
    late = mk_cue([("late", 4.0, 5.0)])
    assert render_caption_track([late], CaptionStyle(), tmp_path, duration=3.0) is None


LONG = "Absolutely everyone should understand compound interest before turning thirty"
HUGE_WORD = "Pneumonoultramicroscopicsilicovolcanoconiosis"


@pytest.mark.parametrize("size,color,position,upper", list(itertools.product(
    ["S", "M", "L"], ["white", "yellow", "boxed"], ["top", "middle", "bottom"], [False, True])))
def test_all_presets_keep_text_in_safe_area(tmp_path: Path, size: str, color: str, position: str,
                                            upper: bool) -> None:
    style = CaptionStyle(size=size, color=color, position=position, uppercase=upper)
    cues = [
        mk_cue([(w, i * 0.2, i * 0.2 + 0.15) for i, w in enumerate(LONG.split())], end=2.0),
        mk_cue([(HUGE_WORD, 2.5, 3.5)], end=3.6, idx=40),
        mk_cue([("Hi", 4.0, 4.5)], end=4.6, idx=50),
    ]
    track = render_caption_track(cues, style, tmp_path, duration=5.0)
    assert track is not None
    assert track.y >= TOP_SAFE and track.y + track.h <= 1920 - BOTTOM_SAFE
    centre = {"top": 0.22, "middle": 0.50, "bottom": 0.70}[position] * 1920
    assert abs(track.y + track.h / 2 - centre) < 0.08 * 1920
    for png in tmp_path.glob("cap_*.png"):
        a = alpha(png)
        ys, xs = np.nonzero(a)
        assert len(xs)
        assert xs.min() >= SIDE_MARGIN and xs.max() < 1080 - SIDE_MARGIN, (png.name, xs.min(), xs.max())
        assert track.y + ys.min() >= TOP_SAFE and track.y + ys.max() < 1920 - BOTTOM_SAFE


def test_long_word_shrinks_then_breaks(tmp_path: Path) -> None:
    style = CaptionStyle(size="L", uppercase=True)
    track = render_caption_track([mk_cue([(HUGE_WORD, 0.0, 1.0)], highlight=False)], style, tmp_path, duration=1.0)
    assert track is not None
    a = alpha(next(tmp_path.glob("cap_*.png")))
    rows = np.nonzero(a.any(axis=1))[0]
    # Broken over several lines rather than overflowing the side margins.
    assert rows.max() - rows.min() > 2 * 84 * 0.7


def test_boxed_preset_draws_translucent_box(tmp_path: Path) -> None:
    cue = mk_cue([("Boxed", 0.0, 1.0)], highlight=False)
    track = render_caption_track([cue], CaptionStyle(color="boxed"), tmp_path, duration=1.0)
    assert track is not None
    rgba = np.asarray(Image.open(next(tmp_path.glob("cap_*.png"))))
    box = (np.abs(rgba[:, :, :3].astype(int) - [0x59, 0x1E, 0x4A]).sum(axis=2) < 6) & (rgba[:, :, 3] == 224)
    assert box.sum() > 1000


def test_lines_needed_measures_rendered_width() -> None:
    assert lines_needed(["short"], CaptionStyle()) == 1
    many = LONG.split() * 2
    assert lines_needed(many, CaptionStyle(size="L", uppercase=True)) > lines_needed(many, CaptionStyle(size="S"))
    assert lines_needed([], CaptionStyle()) == 0
