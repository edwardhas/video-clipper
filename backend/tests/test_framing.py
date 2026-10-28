"""Tests for clipper.framing (faces.py + analyze.py).

* Pure decision/tracking/expression logic is unit-tested with synthetic observations.
* Integration tests run real FFmpeg + real YuNet on small synthetic videos built from the
  public-domain face fixture (tests/fixtures/face_portrait_pd.jpg) with lavfi.
* ``crop_filter`` is verified by rendering an x-encoding gradient through FFmpeg.

Face detection is not active-speaker detection; these tests only check mechanics.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import requires_ffmpeg

from clipper.config import Settings
from clipper.errors import CancelledError
from clipper.framing import analyze
from clipper.framing.analyze import (
    FRAMING_VERSION,
    MAX_SEGMENTS,
    SampleObs,
    analyze_framing,
    build_shots,
    classify_faces,
    crop_filter,
    crop_geometry,
    crop_x_expr,
    detect_cuts,
    median_filter,
    plan_for_edit,
    plan_from_observations,
    simplify_keyframes,
    track_shot,
)
from clipper.framing.faces import MODEL_PATH, FaceBox, FaceDetector, detector_available
from clipper.media.probe import probe
from clipper.models import FramingChoice, FramingPlan, MediaInfo
from clipper.procs import CancelToken, run

FIXTURE_FACE = Path(__file__).parent / "fixtures" / "face_portrait_pd.jpg"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FACE_W = 260                      # overlay width of the portrait in the synthetic 1280x720 videos
CUT_AT = 5.0


def _ffmpeg(*args: str) -> None:
    run([FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-y", *args], timeout=180)


def _media(w: int, h: int, duration: float = 10.0, has_video: bool = True) -> MediaInfo:
    return MediaInfo(duration=duration, size_bytes=1, container="mov,mp4", has_video=has_video, has_audio=True,
                     width=w, height=h, coded_width=w, coded_height=h, fps=25.0)


def _interp(plan: FramingPlan, rel_t: float) -> float:
    ts = [k[0] for k in plan.keyframes]
    xs = [k[1] for k in plan.keyframes]
    return float(np.interp(rel_t, ts, xs))


def _assert_keyframes_valid(plan: FramingPlan) -> None:
    assert plan.keyframes, "crop plans carry at least one keyframe"
    ts = [k[0] for k in plan.keyframes]
    assert ts[0] == 0.0
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert len(plan.keyframes) - 1 <= MAX_SEGMENTS
    max_x = plan.src_w - plan.crop_w
    assert all(0.0 <= x <= max_x for _, x in plan.keyframes)


def _no_speaker_claims(reason: str) -> None:
    assert "speaker" not in reason.lower() and "speaking" not in reason.lower()


# ----------------------------------------------------------------------------- media fixtures
@pytest.fixture(scope="module")
def face_offset() -> tuple[float, float]:
    """Face centre inside the portrait when it is scaled to FACE_W px wide (from YuNet itself)."""
    img = cv2.imread(str(FIXTURE_FACE))
    scaled = cv2.resize(img, (FACE_W, round(img.shape[0] * FACE_W / img.shape[1])), interpolation=cv2.INTER_AREA)
    canvas = np.full((scaled.shape[0] + 80, FACE_W + 200, 3), 40, np.uint8)
    canvas[40:40 + scaled.shape[0], 100:100 + FACE_W] = scaled
    faces = FaceDetector().detect(canvas)
    assert len(faces) == 1
    return faces[0].cx - 100, faces[0].cy - 40


@pytest.fixture(scope="module")
def videos(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("framing-media")
    enc = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p"]
    face = ["-loop", "1", "-framerate", "25", "-i", str(FIXTURE_FACE)]
    out: dict[str, Path] = {}

    # 8 s, 1280x720: face moves right at 60 px/s on a dark background; hard cut at 5 s to a
    # different background where the face jumps to x=880.
    out["moving"] = d / "moving.mp4"
    _ffmpeg("-f", "lavfi", "-i", f"color=c=0x1e2430:s=1280x720:r=25:d={CUT_AT}",
            "-f", "lavfi", "-i", f"color=c=0x8a5a2b:s=1280x720:r=25:d={8 - CUT_AT}", *face,
            "-filter_complex",
            f"[0:v][1:v]concat=n=2:v=1:a=0[bg];[2:v]scale={FACE_W}:-2,setsar=1[f];"
            f"[bg][f]overlay=x='if(lt(t,{CUT_AT}),120+60*t,880)':y=200:shortest=1[v]",
            "-map", "[v]", "-t", "8", *enc, str(out["moving"]))

    # 3 s, two equally sized faces side by side.
    out["two"] = d / "two.mp4"
    _ffmpeg("-f", "lavfi", "-i", "color=c=0x1e2430:s=1280x720:r=25:d=3", *face,
            "-filter_complex",
            f"[1:v]scale={FACE_W}:-2,setsar=1,split[a][b];[0:v][a]overlay=x=150:y=200:shortest=1[t];"
            "[t][b]overlay=x=850:y=200:shortest=1[v]",
            "-map", "[v]", "-t", "3", *enc, str(out["two"]))

    # 3 s test pattern without faces (stand-in for slides / screen recordings).
    out["noface"] = d / "noface.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=s=1280x720:r=25:d=3", "-t", "3", *enc, str(out["noface"]))

    # 3 s vertical 720x1280 source with one face.
    out["vertical"] = d / "vertical.mp4"
    _ffmpeg("-f", "lavfi", "-i", "color=c=0x1e2430:s=720x1280:r=25:d=3", *face,
            "-filter_complex", "[1:v]scale=400:-2,setsar=1[f];[0:v][f]overlay=x='140+20*t':y=300:shortest=1[v]",
            "-map", "[v]", "-t", "3", *enc, str(out["vertical"]))

    # 4 s, 1280x720 lossless gradient whose pixels encode their own column: B = x & 255, G = x >> 8.
    cols = np.arange(1280, dtype=np.uint16)
    row = np.stack([cols & 255, cols >> 8, np.full_like(cols, 77)], axis=-1).astype(np.uint8)
    png = d / "gradient.png"
    cv2.imwrite(str(png), np.repeat(row[None, :, :], 720, axis=0))
    out["gradient"] = d / "gradient.mkv"
    _ffmpeg("-loop", "1", "-framerate", "25", "-i", str(png), "-t", "4", "-c:v", "ffv1", "-pix_fmt", "bgr0",
            str(out["gradient"]))
    png.unlink()

    (d / "garbage.mp4").write_bytes(b"\x00not a video\xff" * 512)
    out["garbage"] = d / "garbage.mp4"
    return out


# ----------------------------------------------------------------------------- faces.py
def test_detector_available_and_missing_or_corrupt_model(tmp_path: Path) -> None:
    assert MODEL_PATH.is_file()
    assert detector_available() is True
    assert detector_available(tmp_path / "missing.onnx") is False
    corrupt = tmp_path / "corrupt.onnx"
    corrupt.write_bytes(b"not an onnx model")
    assert detector_available(corrupt) is False
    with pytest.raises(FileNotFoundError):
        FaceDetector(tmp_path / "missing.onnx")


def test_face_detector_maps_downscaled_boxes_back_to_frame_pixels() -> None:
    img = cv2.imread(str(FIXTURE_FACE))                      # 500x600 portrait
    small = np.full((720, 1280, 3), 30, np.uint8)
    small[60:660, 200:700] = img
    big = cv2.resize(small, (2560, 1440), interpolation=cv2.INTER_LINEAR)   # > 640 wide -> downscaled inside
    det = FaceDetector()
    a, b = det.detect(small), det.detect(big)
    assert len(a) == 1 and len(b) == 1
    assert b[0].score >= 0.6
    assert b[0].cx == pytest.approx(2 * a[0].cx, abs=12)
    assert b[0].cy == pytest.approx(2 * a[0].cy, abs=12)
    assert b[0].w == pytest.approx(2 * a[0].w, rel=0.08)
    assert 200 < a[0].cx < 700                                # inside the pasted portrait
    assert det.detect(np.zeros((360, 640, 3), np.uint8)) == []
    with pytest.raises(ValueError):
        det.detect(np.zeros((32, 32), np.uint8))


# ----------------------------------------------------------------------------- geometry & classification
@pytest.mark.parametrize(("size", "expected"), [
    ((1280, 720), (404, 720, 0)),
    ((1920, 1080), (608, 1080, 0)),
    ((720, 1280), (720, 1280, 0)),          # exactly 9:16: full frame
    ((600, 1280), (600, 1066, 107)),        # narrower than 9:16: full width, centred 16:9-vertical height
    ((641, 360), (202, 360, 0)),
])
def test_crop_geometry(size: tuple[int, int], expected: tuple[int, int, int]) -> None:
    crop_w, crop_h, crop_y = crop_geometry(*size)
    assert (crop_w, crop_h, crop_y) == expected
    assert crop_w % 2 == 0 and crop_w <= size[0] and crop_y + crop_h <= size[1]


def test_sample_width_bounds_the_longer_side() -> None:
    assert analyze.sample_width(1280, 720) == 480
    assert analyze.sample_width(720, 1280) == 270
    assert analyze.sample_width(100, 4000) == 64


def test_classify_faces() -> None:
    big = FaceBox(100, 50, 80, 100, 0.9)
    medium = FaceBox(400, 60, 50, 60, 0.9)        # area 3000 vs 8000 -> ratio 2.67 -> dominant
    similar = FaceBox(400, 60, 70, 90, 0.9)       # area 6300 -> ratio 1.27 -> multi
    tiny = FaceBox(10, 10, 6, 8, 0.9)             # < 5 % of frame height -> ignored
    assert classify_faces([], 270) == ("none", None)
    assert classify_faces([tiny], 270) == ("none", None)
    assert classify_faces([big, tiny], 270) == ("single", big)
    assert classify_faces([medium, big], 270) == ("single", big)
    assert classify_faces([big, similar], 270) == ("multi", None)


def test_cuts_and_shot_majority_weighted_by_duration() -> None:
    assert detect_cuts([0.99, 0.95, 0.2, 0.97, 0.49]) == [3, 5]
    kinds = ["single", "single", "none", "multi", "multi", "single", "none", "none"]
    obs = [SampleObs(src_t=10 + i * 0.5, kind=k) for i, k in enumerate(kinds)]  # type: ignore[arg-type]
    shots = build_shots(obs, [3, 5], src_end=14.5)
    assert [(s.start, s.end, s.kind) for s in shots] == [(0, 3, "single"), (3, 5, "multi"), (5, 8, "none")]
    assert [s.duration for s in shots] == pytest.approx([1.5, 1.0, 2.0])
    tie = build_shots([SampleObs(0.0, "single"), SampleObs(0.5, "multi")], [], src_end=1.0)
    assert tie[0].kind == "multi"                                      # ties resolve conservatively


def test_median_filter_keeps_gaps_and_rejects_outliers() -> None:
    values = [100.0, 102.0, 900.0, 104.0, None, 106.0, 108.0]
    out = median_filter(values, 5)
    assert out[4] is None
    assert out[2] == pytest.approx(103.0)                             # median of 100, 102, 900, 104
    assert out[0] == 100.0 and out[-1] == 108.0                        # centred window shrinks at the edges


def test_track_shot_dead_zone_speed_limit_hold_and_clamp() -> None:
    times = [i * 0.25 for i in range(12)]
    crop_w, src_w = 400, 1280
    # Jitter inside the dead zone (±28 px) -> camera does not move.
    jitter = [600.0 + (10 if i % 2 else -10) for i in range(12)]
    assert len(set(track_shot(times, jitter, crop_w=crop_w, src_w=src_w, default_centre=640))) == 1
    # Big jump -> pans no faster than 0.35 * crop_w per second (35 px per 0.25 s sample).
    jump = [300.0] * 3 + [1000.0] * 9
    centres = track_shot(times, jump, crop_w=crop_w, src_w=src_w, default_centre=640)
    steps = np.abs(np.diff(centres))
    assert steps.max() <= 0.35 * crop_w * 0.25 + 1e-6
    assert centres[-1] > centres[2] + 200
    # Missing detections hold the position; no detections at all -> default centre.
    holes = [500.0, 500.0, None, None, None, None, 500.0, None, None, None, None, None]
    assert set(track_shot(times, holes, crop_w=crop_w, src_w=src_w, default_centre=640)) == {500.0}
    assert set(track_shot(times, [None] * 12, crop_w=crop_w, src_w=src_w, default_centre=640)) == {640.0}
    # Faces at the very edge clamp the window inside the frame.
    edge = track_shot(times, [5.0] * 12, crop_w=crop_w, src_w=src_w, default_centre=640)
    assert set(edge) == {200.0}
    # A cut to a face still inside the dead zone keeps the previous position (no needless jump).
    assert track_shot(times[:3], [610.0] * 3, crop_w=crop_w, src_w=src_w, default_centre=640,
                      start_centre=600.0)[0] == 600.0


def test_simplify_keyframes_bounds_segments_and_preserves_shape() -> None:
    line = [(i * 0.25, 100.0 + 10.0 * i) for i in range(200)]
    assert simplify_keyframes(line, eps=1.0) == [line[0], line[-1]]
    rng = np.random.default_rng(7)
    noisy = [(i * 0.25, float(400 + 300 * np.sin(i / 3.0) + rng.normal(0, 4))) for i in range(500)]
    out = simplify_keyframes(noisy, max_segments=60, eps=1.0)
    assert len(out) - 1 <= 60
    assert out[0] == noisy[0] and out[-1] == noisy[-1]


# ----------------------------------------------------------------------------- decision (pure)
def _obs(kinds: str, *, start: float = 20.0, dt: float = 0.25, cx: float = 700.0) -> list[SampleObs]:
    names = {"s": "single", "m": "multi", "n": "none"}
    return [SampleObs(src_t=start + i * dt, kind=names[k], face_cx=cx if k == "s" else None)  # type: ignore[arg-type]
            for i, k in enumerate(kinds)]


def test_plan_from_observations_decisions_and_reasons() -> None:
    kw = {"src_start": 20.0, "src_w": 1280, "src_h": 720}
    crop = plan_from_observations(_obs("s" * 14 + "n" * 6), [14], src_end=25.0, **kw)
    assert crop.layout == "crop" and crop.version == FRAMING_VERSION and crop.detector == "yunet-2023mar"
    assert (crop.crop_w, crop.crop_h, crop.crop_y) == (404, 720, 0)
    assert crop.stats["share_single"] == pytest.approx(0.7) and crop.stats["shots"] == 2
    assert crop.stats["samples"] == 20 and "70%" in crop.reason
    _assert_keyframes_valid(crop)
    assert _interp(crop, 1.0) == pytest.approx(700 - 202, abs=1)    # follows the face
    assert _interp(crop, 4.5) == pytest.approx(640 - 202, abs=1)    # face-less shot -> centre

    # Per-shot framing (round 2): a face shot next to a >= 1 s shot without a single face is cropped and the
    # other shot becomes a fit span (this was a whole-clip "fit" before). Short alternating shots still
    # decide on the whole clip.
    per_shot = plan_from_observations(_obs("s" * 10 + "n" * 10), [10], src_end=25.0, **kw)
    assert per_shot.layout == "crop" and per_shot.fit_spans == [(2.375, 5.0)]
    mixed = plan_from_observations(_obs("ssnn" * 5), list(range(2, 20, 2)), src_end=25.0, **kw)
    assert mixed.layout == "fit" and "only 50%" in mixed.reason
    # (a sub-second single-face shot after a wide shot is not cropped on its own: still "fit")
    multi = plan_from_observations(_obs("m" * 18 + "s" * 2), [18], src_end=25.0, **kw)
    assert multi.layout == "fit" and multi.reason == analyze.REASON_MULTI
    none = plan_from_observations(_obs("n" * 20), [], src_end=25.0, **kw)
    assert none.layout == "fit" and none.reason == analyze.REASON_NONE
    assert none.stats["share_none"] == 1.0 and none.keyframes == []
    for plan in (crop, per_shot, mixed, multi, none):
        _no_speaker_claims(plan.reason)
        assert plan.src_start == 20.0 and plan.src_end == 25.0


def test_many_cuts_still_yield_bounded_valid_keyframes() -> None:
    obs: list[SampleObs] = []
    cuts: list[int] = []
    for k in range(100):                                      # 100 one-second shots, face alternates sides
        if k:
            cuts.append(len(obs))
        cx = 300.0 if k % 2 else 1000.0
        obs += [SampleObs(src_t=k + j * 0.25, kind="single", face_cx=cx) for j in range(4)]
    plan = plan_from_observations(obs, cuts, src_start=0.0, src_end=100.0, src_w=1280, src_h=720)
    assert plan.layout == "crop" and plan.stats["shots"] == 100
    _assert_keyframes_valid(plan)


# ----------------------------------------------------------------------------- plan_for_edit (pure)
def _auto_crop(src_start: float = 10.0, src_end: float = 20.0) -> FramingPlan:
    return FramingPlan(version=FRAMING_VERSION, layout="crop", src_start=src_start, src_end=src_end, src_w=1280,
                       src_h=720, crop_w=404, crop_h=720, crop_y=0,
                       keyframes=[(0.0, 100.0), (4.0, 500.0), (10.0, 500.0)],
                       reason="A single clear face is on screen for 90% of this clip, so the vertical crop "
                              "follows that face.", detector="yunet-2023mar", stats={"share_single": 0.9})


def test_plan_for_edit_fit_and_manual_crop() -> None:
    media = _media(1280, 720)
    fit = plan_for_edit(_auto_crop(), FramingChoice(mode="fit"), media, 10.0, 20.0)
    assert fit.layout == "fit" and fit.keyframes == [] and fit.detector == "manual"
    centre = plan_for_edit(_auto_crop(), FramingChoice(mode="crop", crop_x=0.5), media, 10.0, 20.0)
    assert centre.layout == "crop" and centre.keyframes == [(0.0, 438.0)] and centre.crop_w == 404
    left = plan_for_edit(None, FramingChoice(mode="crop", crop_x=0.0), media, 3.0, 9.0)
    right = plan_for_edit(None, FramingChoice(mode="crop", crop_x=1.0), media, 3.0, 9.0)
    assert left.keyframes == [(0.0, 0.0)] and right.keyframes == [(0.0, 876.0)]   # clamped to the frame
    assert left.src_start == 3.0 and left.src_end == 9.0
    vertical = plan_for_edit(None, FramingChoice(mode="crop", crop_x=0.9), _media(600, 1280), 0.0, 5.0)
    assert (vertical.crop_w, vertical.crop_h, vertical.crop_y) == (600, 1066, 107)
    assert vertical.keyframes == [(0.0, 0.0)]


def test_plan_for_edit_crop_without_position_and_auto_mode() -> None:
    media = _media(1280, 720)
    auto = _auto_crop()
    tracked = plan_for_edit(auto, FramingChoice(mode="crop"), media, 10.0, 20.0)
    assert tracked.keyframes == auto.keyframes and tracked.layout == "crop"
    # Range nudged within 0.05 s: keyframes re-based so rel times stay tied to source times.
    nudged = plan_for_edit(auto, FramingChoice(mode="auto"), media, 10.04, 20.02)
    assert nudged.src_start == 10.04 and nudged.src_end == 20.02
    assert nudged.keyframes[0][0] == 0.0 and _interp(nudged, 1.96) == pytest.approx(_interp(auto, 2.0), abs=0.2)
    # Range changed: keyframes from another range must never be reused.
    moved = plan_for_edit(auto, FramingChoice(mode="crop"), media, 12.0, 22.0)
    assert moved.keyframes == [(0.0, 438.0)] and moved.reason == analyze.REASON_CENTRE_CROP
    pending = plan_for_edit(auto, FramingChoice(mode="auto"), media, 12.0, 22.0)
    assert pending.layout == "crop" and pending.keyframes == [(0.0, 438.0)]
    assert "re-analysis pending" in pending.reason and pending.src_start == 12.0
    auto_fit = FramingPlan(version=FRAMING_VERSION, layout="fit", src_start=10.0, src_end=20.0, src_w=1280,
                           src_h=720, reason=analyze.REASON_MULTI, detector="yunet-2023mar")
    assert plan_for_edit(auto_fit, FramingChoice(mode="auto"), media, 10.0, 20.0).reason == analyze.REASON_MULTI
    pending_fit = plan_for_edit(auto_fit, FramingChoice(mode="auto"), media, 0.0, 9.0)
    assert pending_fit.layout == "fit" and "re-analysis pending" in pending_fit.reason
    assert plan_for_edit(None, FramingChoice(mode="auto"), media, 0.0, 9.0).layout == "fit"
    centre_no_face = plan_for_edit(auto_fit, FramingChoice(mode="crop"), media, 10.0, 20.0)
    assert centre_no_face.layout == "crop" and centre_no_face.keyframes == [(0.0, 438.0)]
    assert "no single clear face" in centre_no_face.reason


def _cut_track(n_cuts: int = 30) -> list[tuple[float, float]]:
    """``1 + 2 * n_cuts`` keyframes: hard cuts every 2 s alternating x = 100 / 900."""
    points = [(0.0, 100.0)]
    x = 100.0
    for k in range(n_cuts):
        t_cut = 2.0 * k + 1.0
        nxt = 900.0 if x == 100.0 else 100.0
        points += [(round(t_cut - 0.001, 3), x), (round(t_cut + 0.001, 3), nxt)]
        x = nxt
    return points


def _rendered_x(plan: FramingPlan, clip_src_start: float, t: float) -> float:
    from clipper.framing.expr import crop_points, crop_x_expression

    expr = crop_x_expression(crop_points(plan, clip_src_start=clip_src_start))
    env = {"clip": lambda v, lo, hi: min(max(v, lo), hi), "gte": lambda a, b: 1.0 if a >= b else 0.0, "t": t}
    return float(eval(expr, {"__builtins__": {}}, env))  # noqa: S307 - expression built by our own code


def test_rebasing_a_full_track_keeps_every_instant_cut() -> None:
    # Review finding 35: a 61-keyframe auto track re-based to a start 0.03 s earlier got a 62nd
    # (head) point; crop_points then decimated by index and could turn a hard cut into a 2 s pan.
    track = _cut_track()
    assert len(track) == MAX_SEGMENTS + 1
    auto = FramingPlan(version=FRAMING_VERSION, layout="crop", src_start=100.03, src_end=170.0, src_w=1920,
                       src_h=1080, crop_w=608, crop_h=1080, crop_y=0, keyframes=track, reason="x",
                       detector="yunet-2023mar")
    plan = plan_for_edit(auto, FramingChoice(mode="auto"), _media(1920, 1080, duration=200.0), 100.0, 170.0)
    assert plan.src_start == 100.0 and len(plan.keyframes) <= MAX_SEGMENTS + 1
    for k in range(30):
        t_cut = 2.0 * k + 1.0 + 0.03                          # clip-relative after the re-base
        before, after = (100.0, 900.0) if k % 2 == 0 else (900.0, 100.0)
        assert _rendered_x(plan, 100.0, t_cut - 0.05) == pytest.approx(before, abs=1.0), k
        assert _rendered_x(plan, 100.0, t_cut + 0.05) == pytest.approx(after, abs=1.0), k


def test_crop_points_reduce_long_tracks_by_shape_not_index() -> None:
    from clipper.framing.expr import crop_points

    track = [(0.0, 100.0), (0.5, 100.0)] + [(t + 0.5, x) for t, x in _cut_track()[1:]]   # 62 points
    plan = FramingPlan(version=FRAMING_VERSION, layout="crop", src_start=0.0, src_end=70.0, src_w=1920,
                       src_h=1080, crop_w=608, crop_h=1080, crop_y=0, keyframes=track, reason="x", detector="x")
    pts = crop_points(plan, clip_src_start=0.0)
    assert len(pts) <= MAX_SEGMENTS + 1
    for t in [0.2, 1.4, 1.6, 3.4, 3.6, 30.4, 30.6, 59.4, 59.6, 61.0]:
        assert _rendered_x(plan, 0.0, t) == pytest.approx(_interp(plan, t), abs=1.0), t


# ----------------------------------------------------------------------------- crop_x_expr (pure)
def test_crop_x_expr_format() -> None:
    plan = _auto_crop()
    expr = crop_x_expr(plan)
    assert expr.startswith("clip(") and expr.endswith(",0,876)")
    assert "e-" not in expr and "e+" not in expr and "'" not in expr and ":" not in expr
    assert "if(" not in expr and "clip(t-" in expr  # flat sum of clipped linear terms (no nesting)
    static = plan.model_copy(update={"keyframes": [(0.0, 1234.0)]})
    assert crop_x_expr(static) == "876.00"           # single keyframe -> constant, clamped
    assert crop_filter(static) == "crop=w=404:h=720:x='876.00':y=0"
    fit = FramingPlan(version=1, layout="fit", src_start=0, src_end=1, src_w=1280, src_h=720, reason="x",
                      detector="none")
    with pytest.raises(ValueError):
        crop_x_expr(fit)
    many = plan.model_copy(update={"keyframes": [(i * 0.1, float(100 + 700 * (i % 2))) for i in range(200)]})
    assert len(re.findall(r"clip\(t-|gte\(t,", crop_x_expr(many))) <= MAX_SEGMENTS


# ----------------------------------------------------------------------------- crop_filter through FFmpeg
def _render_columns(src: Path, plan: FramingPlan, *, seek: float, duration: float, settings: Settings) -> list[int]:
    """Render ``src`` through ``crop_filter(plan)``; return the source column of each frame's left edge."""
    w, h = plan.crop_w, plan.crop_h
    chunks: list[bytes] = []
    run([settings.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-ss", f"{seek:.3f}", "-i", str(src),
         "-t", f"{duration:.3f}", "-vf", f"{crop_filter(plan)},format=bgr24", "-f", "rawvideo", "-"],
        stdout_bytes_cb=chunks.append, timeout=120)
    raw = b"".join(chunks)
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return [int(f[h // 2, 0, 0]) + 256 * int(f[h // 2, 0, 1]) for f in frames]


@pytest.mark.ffmpeg
@requires_ffmpeg
@pytest.mark.parametrize("shape", ["jump", "zigzag60"])
def test_crop_filter_moves_window_as_planned(videos: dict[str, Path], settings: Settings, shape: str) -> None:
    if shape == "jump":
        keyframes = [(0.0, 0.0), (1.2, 600.0), (1.202, 100.0), (2.5, 876.0)]
    else:   # MAX_SEGMENTS segments: deepest expression the plan builder can produce
        keyframes = [(i * 0.05, float(50 + 700 * (i % 2))) for i in range(MAX_SEGMENTS + 1)]
    plan = FramingPlan(version=FRAMING_VERSION, layout="crop", src_start=1.0, src_end=4.0, src_w=1280, src_h=720,
                       crop_w=404, crop_h=720, keyframes=keyframes, reason="test", detector="none")
    # The renderer seeks with -ss before -i, so t restarts at 0 for the clip.
    columns = _render_columns(videos["gradient"], plan, seek=1.0, duration=3.0, settings=settings)
    assert len(columns) == 75
    ts = [k[0] for k in keyframes]
    xs = [k[1] for k in keyframes]
    for i, column in enumerate(columns):
        expected = float(np.clip(np.interp(i / 25.0, ts, xs), 0, 876))
        assert abs(column - expected) <= 1.0, (i, column, expected)


# ----------------------------------------------------------------------------- analyze_framing (real ffmpeg + YuNet)
@pytest.mark.ffmpeg
@requires_ffmpeg
def test_moving_face_with_cut_is_cropped_and_tracked(videos: dict[str, Path], settings: Settings,
                                                     face_offset: tuple[float, float]) -> None:
    media = probe(videos["moving"], settings=settings)
    plan = analyze_framing(videos["moving"], media, 0.0, 8.0, settings=settings, cancel=None)
    assert plan.layout == "crop", plan.reason
    assert (plan.crop_w, plan.crop_h, plan.crop_y) == (404, 720, 0)
    assert plan.detector == "yunet-2023mar" and plan.stats["shots"] == 2
    assert plan.stats["share_single"] >= 0.9 and plan.stats["samples"] == 32
    _assert_keyframes_valid(plan)
    _no_speaker_claims(plan.reason)

    def expected_left(rel_t: float) -> float:
        face_x = (120 + 60 * rel_t) if rel_t < CUT_AT else 880.0
        return float(np.clip(face_x + face_offset[0] - plan.crop_w / 2, 0, plan.src_w - plan.crop_w))

    tolerance = 0.07 * plan.crop_w + 20       # dead zone + detection/encoding noise
    for rel_t in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5):
        assert abs(_interp(plan, rel_t) - expected_left(rel_t)) <= tolerance, rel_t
    # Instant jump at the cut (between the 4.75 s and 5.0 s samples), no slow pan.
    before, after = _interp(plan, 4.8), _interp(plan, 4.95)
    assert after - before > 400
    assert abs(after - expected_left(5.5)) <= tolerance


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_sub_range_keyframes_are_clip_relative(videos: dict[str, Path], settings: Settings,
                                               face_offset: tuple[float, float]) -> None:
    media = probe(videos["moving"], settings=settings)
    plan = analyze_framing(videos["moving"], media, 2.0, 4.5, settings=settings, cancel=None)
    assert plan.layout == "crop" and plan.src_start == 2.0 and plan.src_end == 4.5
    assert plan.stats["shots"] == 1 and plan.stats["samples"] == 10
    _assert_keyframes_valid(plan)
    for rel_t in (0.25, 1.25, 2.25):
        src_t = 2.0 + rel_t
        expected = 120 + 60 * src_t + face_offset[0] - plan.crop_w / 2
        assert abs(_interp(plan, rel_t) - expected) <= 0.07 * plan.crop_w + 20


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_two_faces_keep_full_frame(videos: dict[str, Path], settings: Settings) -> None:
    media = probe(videos["two"], settings=settings)
    plan = analyze_framing(videos["two"], media, 0.0, 3.0, settings=settings, cancel=None)
    assert plan.layout == "fit"
    assert plan.reason == "Multiple people are on screen, so the full frame is kept."
    assert plan.stats["share_multi"] >= 0.9 and plan.keyframes == []
    _no_speaker_claims(plan.reason)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_no_faces_keep_full_frame(videos: dict[str, Path], settings: Settings) -> None:
    media = probe(videos["noface"], settings=settings)
    plan = analyze_framing(videos["noface"], media, 0.0, 3.0, settings=settings, cancel=None)
    assert plan.layout == "fit"
    assert plan.reason == "No clear face found (e.g. slides), so the full frame is kept."
    assert plan.stats["share_none"] >= 0.9


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_vertical_source_uses_full_width_crop(videos: dict[str, Path], settings: Settings) -> None:
    media = probe(videos["vertical"], settings=settings)
    assert (media.width, media.height) == (720, 1280)
    plan = analyze_framing(videos["vertical"], media, 0.0, 3.0, settings=settings, cancel=None)
    assert plan.layout == "crop", plan.reason
    assert (plan.crop_w, plan.crop_h, plan.crop_y) == (720, 1280, 0)
    assert plan.keyframes == [(0.0, 0.0)]
    assert crop_filter(plan) == "crop=w=720:h=1280:x='0':y=0"


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_degraded_paths(videos: dict[str, Path], settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    media = _media(1280, 720, duration=3.0)
    unreadable = analyze_framing(videos["garbage"], media, 0.0, 3.0, settings=settings, cancel=None)
    assert unreadable.layout == "fit" and unreadable.reason == analyze.REASON_UNREADABLE
    assert str(videos["garbage"].parent) not in unreadable.reason
    no_video = analyze_framing(videos["noface"], _media(0, 0, has_video=False), 0.0, 3.0, settings=settings,
                               cancel=None)
    assert no_video.layout == "fit" and no_video.reason == analyze.REASON_NO_VIDEO
    with pytest.raises(ValueError):
        analyze_framing(videos["noface"], media, 2.0, 1.0, settings=settings, cancel=None)
    token = CancelToken()
    token.cancel()
    with pytest.raises(CancelledError):
        analyze_framing(videos["noface"], media, 0.0, 3.0, settings=settings, cancel=token)
    monkeypatch.setattr(analyze, "detector_available", lambda: False)
    missing = analyze_framing(videos["moving"], media, 0.0, 3.0, settings=settings, cancel=None)
    assert missing.layout == "fit" and missing.detector == "none"
    assert missing.reason.startswith("Face detector unavailable")
