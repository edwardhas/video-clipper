"""Per-shot framing (round-2 work item B): shots without a single clear face inside a crop clip are shown
full-frame (``FramingPlan.fit_spans``), so a face crop never cuts a slide's text.

* pure: span building (≥ 1 s runs, merging, gap filling), the crop decision and ``plan_for_edit`` re-basing;
* real FFmpeg + YuNet: a face shot followed by a slide shot is analysed into a crop plan with one fit span
  and rendered: the face is cropped to fill the frame, the slide is letterboxed on a blurred fill, and
  captions are composited on top of both.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest
from conftest import requires_ffmpeg

from clipper.config import Settings
from clipper.framing import analyze
from clipper.framing.analyze import (
    FRAMING_VERSION,
    SampleObs,
    analyze_framing,
    plan_for_edit,
    plan_from_observations,
)
from clipper.media.encoders import cpu_encoder
from clipper.media.probe import probe
from clipper.models import CaptionWord, ClipEdit, Cue, FramingChoice, FramingPlan, MediaInfo
from clipper.procs import run
from clipper.render import renderer
from clipper.render.caption_image import render_caption_track
from clipper.render.renderer import render_clip, render_hash

FIXTURE_FACE = Path(__file__).parent / "fixtures" / "face_portrait_pd.jpg"
KW = {"src_start": 20.0, "src_w": 1280, "src_h": 720}


def _obs(kinds: str, *, start: float = 20.0, dt: float = 0.25, cx: float = 700.0) -> list[SampleObs]:
    names = {"s": "single", "m": "multi", "n": "none"}
    return [SampleObs(src_t=start + i * dt, kind=names[k], face_cx=cx if k == "s" else None)  # type: ignore[arg-type]
            for i, k in enumerate(kinds)]


def _plan(kinds: str, cuts: list[int]) -> FramingPlan:
    return plan_from_observations(_obs(kinds), cuts, src_end=20.0 + 0.25 * len(kinds), **KW)


def _media(w: int = 1280, h: int = 720) -> MediaInfo:
    return MediaInfo(duration=60.0, size_bytes=1, container="mov,mp4", has_video=True, has_audio=True,
                     width=w, height=h, coded_width=w, coded_height=h, fps=25.0)


# ----------------------------------------------------------------------------- decision (pure)
def test_slide_shots_in_a_face_clip_become_fit_spans() -> None:
    assert FRAMING_VERSION == 3
    lecture = _plan("s" * 20 + "n" * 8 + "s" * 4, [20, 28])               # 75 % face, a 2 s slide shot
    assert lecture.layout == "crop" and lecture.fit_spans == [(4.875, 6.875)]
    assert lecture.reason == ("A single clear face is on screen for 75% of this clip, so the vertical crop follows "
                              "that face; shots without a single clear face (for example slides) keep the full "
                              "frame.")
    assert "speaker" not in lecture.reason.lower()
    # Half face, half slides: the slides are shown full-frame anyway, so the face shot is cropped.
    half = _plan("s" * 16 + "n" * 16, [16])
    assert half.layout == "crop" and half.fit_spans == [(3.875, 8.0)] and "50%" in half.reason
    assert half.keyframes[0] == (0.0, pytest.approx(700 - 202, abs=1))
    # Several people (a wide shot) are not cropped either.
    wide = _plan("s" * 16 + "m" * 8 + "s" * 8, [16, 24])
    assert wide.layout == "crop" and wide.fit_spans == [(3.875, 5.875)]


def test_fit_spans_need_one_second_and_never_flicker() -> None:
    # A slide shot shorter than 1 s stays cropped (and the plan is the plain crop it always was).
    short = _plan("s" * 16 + "n" * 3 + "s" * 13, [16, 19])
    assert short.layout == "crop" and short.fit_spans == [] and short.reason == analyze.REASON_CROP.format(pct=91)
    # Consecutive short shots without a single face form one run (fast slide changes).
    merged = _plan("s" * 16 + "n" * 3 + "m" * 3 + "n" * 2 + "s" * 16, [16, 19, 22, 24])
    assert merged.fit_spans == [(3.875, 5.875)]
    # A face shot shorter than 1 s between two spans, or at a clip edge, is not cropped on its own.
    gap = _plan("s" * 12 + "n" * 8 + "s" * 3 + "n" * 8 + "s" * 12, [12, 20, 23, 31])
    assert gap.layout == "crop" and gap.fit_spans == [(2.875, 7.625)]
    tail = _plan("s" * 24 + "n" * 8 + "s" * 3, [24, 32])
    assert tail.fit_spans == [(5.875, 8.75)]
    head = _plan("s" * 3 + "n" * 8 + "s" * 24, [3, 11])
    assert head.fit_spans == [(0.0, 2.625)]


def test_fit_layout_when_too_little_would_be_cropped() -> None:
    # Nearly all slides with a sub-second face shot at the end: the whole clip is full-frame.
    slides = _plan("n" * 20 + "s" * 3, [20])
    assert slides.layout == "fit" and slides.fit_spans == [] and slides.reason == analyze.REASON_NONE
    # Short alternating shots (no span qualifies) are still judged on the whole clip.
    mixed = _plan("ssnn" * 5, [i for i in range(2, 20, 2)])
    assert mixed.layout == "fit" and "only 50%" in mixed.reason
    for plan in (slides, mixed):
        assert plan.fit_spans == []


# ----------------------------------------------------------------------------- plan_for_edit (pure)
def _auto(**kw: object) -> FramingPlan:
    data: dict[str, object] = dict(version=FRAMING_VERSION, layout="crop", src_start=10.0, src_end=20.0, src_w=1280,
                                   src_h=720, crop_w=404, crop_h=720, crop_y=0,
                                   keyframes=[(0.0, 100.0), (4.0, 500.0), (10.0, 500.0)],
                                   fit_spans=[(4.0, 6.0), (8.5, 10.0)],
                                   reason=analyze.REASON_CROP_SPANS.format(pct=70), detector="yunet-2023mar")
    data.update(kw)
    return FramingPlan(**data)  # type: ignore[arg-type]


def test_auto_mode_keeps_and_rebases_fit_spans() -> None:
    media, auto = _media(), _auto()
    same = plan_for_edit(auto, FramingChoice(mode="auto"), media, 10.0, 20.0)
    assert same.fit_spans == auto.fit_spans and same.reason == auto.reason
    later = plan_for_edit(auto, FramingChoice(mode="auto"), media, 10.04, 20.02)
    # Shifted with the keyframes; the last span still runs to the (new) clip end.
    assert later.fit_spans == [(3.96, 5.96), (8.46, 9.98)]
    earlier = plan_for_edit(auto, FramingChoice(mode="auto"), media, 9.96, 20.04)
    assert earlier.fit_spans == [(4.04, 6.04), (8.54, 10.08)]
    starts_fit = _auto(fit_spans=[(0.0, 2.0)])
    assert plan_for_edit(starts_fit, FramingChoice(mode="auto"), media, 9.97, 20.0).fit_spans == [(0.0, 2.03)]
    assert plan_for_edit(starts_fit, FramingChoice(mode="auto"), media, 10.03, 20.0).fit_spans == [(0.0, 1.97)]
    tiny = _auto(fit_spans=[(0.0, 0.03), (5.0, 6.0)])
    assert plan_for_edit(tiny, FramingChoice(mode="auto"), media, 10.04, 20.0).fit_spans == [(4.96, 5.96)]
    # Another range: nothing analysed for it is reused (centre crop pending, no spans).
    pending = plan_for_edit(auto, FramingChoice(mode="auto"), media, 12.0, 22.0)
    assert pending.fit_spans == [] and "re-analysis pending" in pending.reason


def test_manual_framing_has_no_fit_spans() -> None:
    media, auto = _media(), _auto()
    tracked = plan_for_edit(auto, FramingChoice(mode="crop"), media, 10.0, 20.0)
    assert tracked.layout == "crop" and tracked.keyframes == auto.keyframes and tracked.fit_spans == []
    assert tracked.reason == analyze.REASON_MANUAL_CROP_ALL and "speaker" not in tracked.reason.lower()
    placed = plan_for_edit(auto, FramingChoice(mode="crop", crop_x=0.3), media, 10.0, 20.0)
    assert placed.layout == "crop" and placed.fit_spans == []
    fit = plan_for_edit(auto, FramingChoice(mode="fit"), media, 10.0, 20.0)
    assert fit.layout == "fit" and fit.fit_spans == []


def test_fit_spans_are_part_of_the_render_hash() -> None:
    assert renderer.RENDER_VERSION == 3
    edit = ClipEdit(src_start=10.0, src_end=20.0, captions_enabled=False)
    hashes = {render_hash(source_sha256="ab" * 32, edit=edit, plan=_auto(fit_spans=spans), cues=[],
                          encoder="libx264")
              for spans in ([], [(4.0, 6.0)], [(4.0, 6.5)])}
    assert len(hashes) == 3


# ----------------------------------------------------------------------------- real FFmpeg + YuNet
def _ffmpeg(*args: str) -> None:
    run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", *args], timeout=180)


def _frame(path: Path, t: float, size: tuple[int, int]) -> np.ndarray:
    res = run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
               "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], timeout=60)
    return np.frombuffer(res.stdout, np.uint8).reshape(size[1], size[0], 3)


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference after downscaling both to 108 px wide (robust to resampling/encoding)."""
    h = round(a.shape[0] * 108 / a.shape[1])
    small = [cv2.resize(np.ascontiguousarray(x), (108, h), interpolation=cv2.INTER_AREA).astype(float) for x in (a, b)]
    return float(np.abs(small[0] - small[1]).mean())


def _expected_crop(src_frame: np.ndarray, plan: FramingPlan, rel_t: float) -> np.ndarray:
    ts, xs = [k[0] for k in plan.keyframes], [k[1] for k in plan.keyframes]
    x = int(round(float(np.interp(rel_t, ts, xs))))
    return cv2.resize(src_frame[:, x:x + plan.crop_w], (1080, 1920), interpolation=cv2.INTER_AREA)


def _expected_fit_band(src_frame: np.ndarray) -> np.ndarray:
    """The sharp, letterboxed foreground of the fit layout (rows 656..1264 of the output)."""
    return cv2.resize(src_frame, (1080, 608), interpolation=cv2.INTER_AREA)


FIT_ROWS = slice(656 + 8, 656 + 608 - 8)


@pytest.fixture(scope="module")
def lecture(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """8 s 1280x720 at 25 fps: one face on a dark background for 4 s, then a static slide (white with
    black "text lines" across the width) for 4 s; a tone as audio. One keyframe at 0 (and the cut)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("lecture")
    lines = ",".join(f"drawbox=x=60:y={y}:w=1160:h=36:color=black:t=fill" for y in (140, 260, 380, 500))
    src = d / "lecture.mp4"
    _ffmpeg("-f", "lavfi", "-i", "color=c=0x1e2430:s=1280x720:r=25:d=4",
            "-f", "lavfi", "-i", f"color=c=white:s=1280x720:r=25:d=4,{lines}",
            "-loop", "1", "-framerate", "25", "-i", str(FIXTURE_FACE), "-f", "lavfi", "-i", "sine=d=8",
            "-filter_complex", "[2:v]scale=260:-2,setsar=1[f];[0:v][f]overlay=x=520:y=180:shortest=1[face];"
                               "[face][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-map", "3:a", "-t", "8", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "aac", str(src))
    yield src
    shutil.rmtree(d, ignore_errors=True)


def _cue(text: str, a: float, b: float, idx: int) -> Cue:
    wid = f"w{idx:06d}"
    return Cue(id=wid, first_word_id=wid, last_word_id=wid, text=text,
               words=[CaptionWord(word_id=wid, text=text, rel_start=a, rel_end=b)], rel_start=a, rel_end=b,
               highlight=False)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_face_then_slide_is_cropped_then_letterboxed(lecture: Path, settings: Settings, tmp_path: Path) -> None:
    media = probe(lecture, settings=settings)
    auto = analyze_framing(lecture, media, 0.0, 8.0, settings=settings, cancel=None)
    assert auto.layout == "crop", auto.reason
    assert len(auto.fit_spans) == 1
    start, end = auto.fit_spans[0]
    assert start == pytest.approx(4.0, abs=0.25) and end == 8.0
    assert auto.reason == analyze.REASON_CROP_SPANS.format(pct=50)

    edit = ClipEdit(src_start=0.0, src_end=8.0)
    plan = plan_for_edit(auto, edit.framing, media, 0.0, 8.0)
    assert plan.fit_spans == auto.fit_spans
    cues = [_cue("During", 5.0, 5.8, 1)]
    out = tmp_path / "out" / "clip.mp4"
    info = render_clip(src=lecture, media=media, edit=edit, plan=plan, cues=cues, out_mp4=out,
                       work_dir=tmp_path / "work", settings=settings, cancel=None, encoder=cpu_encoder(settings))
    assert info["layout"] == "crop" and info["duration"] == pytest.approx(8.0, abs=0.1)

    face_src, slide_src = _frame(lecture, 2.0, (1280, 720)), _frame(lecture, 6.5, (1280, 720))
    at2, at65 = _frame(out, 2.0, (1080, 1920)), _frame(out, 6.5, (1080, 1920))
    # 2 s: the face shot is cropped and fills the 9:16 frame (not letterboxed).
    assert _diff(at2, _expected_crop(face_src, plan, 2.0)) < 8
    assert _diff(at2[FIT_ROWS], _expected_fit_band(face_src)[8:-8]) > 15
    # 6.5 s: the whole slide (every text line end to end) is letterboxed over a blurred fill.
    assert _diff(at65[FIT_ROWS], _expected_fit_band(slide_src)[8:-8]) < 8
    assert _diff(at65, _expected_crop(slide_src, plan, 6.5)) > 15
    top = at65[:600].astype(float).mean(axis=2)
    assert top.min() > 20                      # blurred fill: no sharp black text lines (cropped: 0)
    # The caption (5.0-5.8 s) is composited on top of the fit picture.
    band = render_caption_track(cues, edit.style, tmp_path / "band", duration=8.0)
    assert band is not None
    rows = slice(band.y, band.y + band.h)
    at54 = _frame(out, 5.4, (1080, 1920))
    assert _diff(at54[rows], at65[rows]) > 3
    assert _diff(at54[:band.y], at65[:band.y]) < 2         # static slide: only the caption differs


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_fit_spans_use_output_time_for_a_clip_that_starts_mid_gop(lecture: Path, settings: Settings,
                                                                  tmp_path: Path) -> None:
    media = probe(lecture, settings=settings)
    auto = analyze_framing(lecture, media, 1.5, 7.5, settings=settings, cancel=None)
    assert auto.layout == "crop" and len(auto.fit_spans) == 1
    start, end = auto.fit_spans[0]
    assert start == pytest.approx(2.5, abs=0.25) and end == 6.0           # clip-relative, not source seconds
    edit = ClipEdit(src_start=1.5, src_end=7.5, captions_enabled=False)
    plan = plan_for_edit(auto, edit.framing, media, 1.5, 7.5)
    out = tmp_path / "sub" / "clip.mp4"
    render_clip(src=lecture, media=media, edit=edit, plan=plan, cues=[], out_mp4=out, work_dir=tmp_path / "w",
                settings=settings, cancel=None, encoder=cpu_encoder(settings))
    # rel 2.1 = src 3.6 (face, before the span) and rel 2.7 = src 4.2 (slide, inside it). A filter time that
    # still counted from the source start (or the keyframe at 0) would put rel 2.1 inside the span too.
    face_src, slide_src = _frame(lecture, 3.6, (1280, 720)), _frame(lecture, 4.2, (1280, 720))
    early, late = _frame(out, 2.1, (1080, 1920)), _frame(out, 2.7, (1080, 1920))
    assert _diff(early, _expected_crop(face_src, plan, 2.1)) < 8
    assert _diff(late[FIT_ROWS], _expected_fit_band(slide_src)[8:-8]) < 8
