"""Tests for clipper.media.frames: JPEG stills and rawvideo frame sampling.

Pure helpers are unit-tested; extraction/sampling run real FFmpeg on tiny synthetic
lavfi media (black for 0–2 s then white for 2–4 s, so frame content encodes time).
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import requires_ffmpeg

from clipper.config import Settings
from clipper.errors import CancelledError, MediaError
from clipper.media.frames import (
    FrameCollector,
    display_size,
    effective_fps,
    extract_frame,
    extract_frames,
    rotation_from_stream,
    sample_frames,
    scaled_size,
)
from clipper.procs import CancelToken, run

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def _ffmpeg(*args: str) -> None:
    run([FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-y", *args], timeout=120)


@pytest.fixture(scope="module")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("frames-media")
    # 320x240 @ 25 fps, 4 s: black until 2.0 s, white afterwards.
    _ffmpeg("-f", "lavfi", "-i", "color=c=black:s=320x240:r=25:d=2",
            "-f", "lavfi", "-i", "color=c=white:s=320x240:r=25:d=2",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p[v]", "-map", "[v]",
            "-c:v", "libx264", "-preset", "ultrafast", "-g", "25", str(d / "bw.mp4"))
    # Same pixels stored 320x240 but flagged for 90° display rotation -> displays 240x320.
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=2", "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", str(d / "plain.mp4"))
    _ffmpeg("-display_rotation", "90", "-i", str(d / "plain.mp4"), "-c", "copy", str(d / "rotated.mp4"))
    (d / "garbage.mp4").write_bytes(b"\x00not a video\xff" * 512)
    return d


def _mean(img: np.ndarray) -> float:
    return float(np.asarray(img, dtype=np.float64).mean())


# ----------------------------------------------------------------------------- pure helpers
def test_scaled_size_keeps_aspect_even_and_never_upscales() -> None:
    assert scaled_size(1280, 720, 480) == (480, 270)
    assert scaled_size(1920, 1080, 640) == (640, 360)
    assert scaled_size(320, 240, 640) == (320, 240)          # no upscaling
    assert scaled_size(721, 405, 1000) == (720, 404)          # odd source -> even output
    w, h = scaled_size(240, 320, 120)                         # portrait
    assert (w, h) == (120, 160)
    for w, h in [scaled_size(1281, 721, 333), scaled_size(99, 1001, 50)]:
        assert w % 2 == 0 and h % 2 == 0
    with pytest.raises(ValueError):
        scaled_size(0, 720, 480)


@pytest.mark.parametrize(("stream", "expected"), [
    ({"side_data_list": [{"rotation": -90}]}, 270),
    ({"side_data_list": [{"rotation": 90}]}, 90),
    ({"side_data_list": [{"rotation": 180}]}, 180),
    ({"side_data_list": [{"side_data_type": "other"}], "tags": {"rotate": "90"}}, 90),
    ({"tags": {"rotate": "270"}}, 270),
    ({"tags": {"rotate": "bogus"}}, 0),
    ({}, 0),
])
def test_rotation_from_stream(stream: dict, expected: int) -> None:
    assert rotation_from_stream(stream) == expected


def test_effective_fps_respects_frame_budget() -> None:
    assert effective_fps(4.0, 10.0, 400) == 4
    rate = effective_fps(4.0, 120.0, 240)
    assert float(rate) == pytest.approx(2.0)
    for fps, duration, budget in [(25.0, 2.0, 7), (4.0, 179.9, 240), (30.0, 3.3, 11), (4.0, 0.7, 1)]:
        rate = effective_fps(fps, duration, budget)
        assert 0 < float(rate) <= fps
        assert math.ceil(duration * rate) <= budget
    with pytest.raises(ValueError):
        effective_fps(0.0, 10.0, 10)
    with pytest.raises(ValueError):
        effective_fps(4.0, 10.0, 0)


def test_frame_collector_splits_stream_across_chunk_boundaries() -> None:
    w, h = 4, 2
    frame_bytes = w * h * 3
    payload = b"".join(bytes([i]) * frame_bytes for i in range(5))
    collector = FrameCollector(w, h, limit=3)
    for start in range(0, len(payload), 7):                  # chunk size unrelated to frame size
        collector.feed(payload[start:start + 7])
    assert len(collector.frames) == 3
    assert collector.dropped == 2
    assert collector.leftover == 0
    for i, frame in enumerate(collector.frames):
        assert frame.shape == (h, w, 3) and frame.dtype == np.uint8
        assert np.all(frame == i)
    collector.feed(b"\x01\x02")
    assert collector.leftover == 2


# ----------------------------------------------------------------------------- JPEG stills (real ffmpeg)
@pytest.mark.ffmpeg
@requires_ffmpeg
def test_extract_frame_accurate_seek_and_size(media_dir: Path, settings: Settings, tmp_path: Path) -> None:
    dark = extract_frame(media_dir / "bw.mp4", 1.5, tmp_path / "dark.jpg", max_width=160, settings=settings)
    bright = extract_frame(media_dir / "bw.mp4", 2.5, tmp_path / "bright.jpg", max_width=160, settings=settings)
    img_dark, img_bright = cv2.imread(str(dark)), cv2.imread(str(bright))
    assert img_dark.shape == (120, 160, 3)
    assert _mean(img_dark) < 30 and _mean(img_bright) > 220
    # Just after the change: accurate seek (not the previous keyframe).
    just_after = extract_frame(media_dir / "bw.mp4", 2.1, tmp_path / "after.jpg", settings=settings)
    assert _mean(cv2.imread(str(just_after))) > 220
    # Never upscaled beyond the source width.
    full = extract_frame(media_dir / "bw.mp4", 0.5, tmp_path / "full.jpg", max_width=1000, settings=settings)
    assert cv2.imread(str(full)).shape == (240, 320, 3)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["after.jpg", "bright.jpg", "dark.jpg", "full.jpg"]


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_extract_frame_rotated_source_is_display_oriented(media_dir: Path, settings: Settings,
                                                         tmp_path: Path) -> None:
    assert display_size(media_dir / "rotated.mp4", settings=settings) == (240, 320)
    out = extract_frame(media_dir / "rotated.mp4", 1.0, tmp_path / "rot.jpg", settings=settings)
    assert cv2.imread(str(out)).shape == (320, 240, 3)
    small = extract_frame(media_dir / "rotated.mp4", 1.0, tmp_path / "rot_small.jpg", max_width=120,
                          settings=settings)
    assert cv2.imread(str(small)).shape == (160, 120, 3)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_extract_frame_failure_leaves_nothing_behind(media_dir: Path, settings: Settings, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    with pytest.raises(MediaError) as info:
        extract_frame(media_dir / "garbage.mp4", 0.5, out_dir / "x.jpg", settings=settings)
    assert str(tmp_path) not in info.value.user_message and "/" not in info.value.user_message
    with pytest.raises(MediaError):
        extract_frame(media_dir / "bw.mp4", 60.0, out_dir / "late.jpg", settings=settings)  # past the end
    assert list(out_dir.iterdir()) == []                      # no partial or temp files


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_extract_frames_order_and_names(media_dir: Path, settings: Settings, tmp_path: Path) -> None:
    times = [2.5, 0.5, 3.5]
    paths = extract_frames(media_dir / "bw.mp4", times, tmp_path / "f", settings=settings)
    assert [p.name for p in paths] == ["t000002500.jpg", "t000000500.jpg", "t000003500.jpg"]
    assert [_mean(cv2.imread(str(p))) > 128 for p in paths] == [True, False, True]
    named = extract_frames(media_dir / "bw.mp4", [1.0, 3.0], tmp_path / "c", max_width=64, settings=settings,
                           prefix="c001")
    assert [p.name for p in named] == ["c001_0.jpg", "c001_1.jpg"]
    assert cv2.imread(str(named[0])).shape == (48, 64, 3)
    token = CancelToken()
    token.cancel()
    with pytest.raises(CancelledError):
        extract_frames(media_dir / "bw.mp4", [1.0], tmp_path / "x", settings=settings, cancel=token)


# ----------------------------------------------------------------------------- raw sampling (real ffmpeg)
@pytest.mark.ffmpeg
@requires_ffmpeg
def test_sample_frames_count_times_shape_and_content(media_dir: Path, settings: Settings) -> None:
    frames, times = sample_frames(media_dir / "bw.mp4", 0.0, 4.0, fps=4.0, width=160, settings=settings,
                                  cancel=None)
    assert len(frames) == 16
    assert times == pytest.approx([i * 0.25 for i in range(16)])
    assert all(f.shape == (120, 160, 3) and f.dtype == np.uint8 for f in frames)
    for frame, t in zip(frames, times):
        if t <= 1.8:
            assert _mean(frame) < 30, t
        elif t >= 2.2:
            assert _mean(frame) > 220, t


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_sample_frames_offset_range_uses_source_times(media_dir: Path, settings: Settings) -> None:
    frames, times = sample_frames(media_dir / "bw.mp4", 1.0, 3.0, fps=4.0, width=80, settings=settings,
                                  cancel=None, display_size_hint=(320, 240))
    assert times == pytest.approx([1.0 + i * 0.25 for i in range(8)])
    assert frames[0].shape == (60, 80, 3)
    assert _mean(frames[times.index(1.75)]) < 30
    assert _mean(frames[times.index(2.25)]) > 220


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_sample_frames_lowers_rate_to_respect_max_frames(media_dir: Path, settings: Settings) -> None:
    frames, times = sample_frames(media_dir / "bw.mp4", 0.0, 4.0, fps=25.0, width=64, settings=settings,
                                  cancel=None, max_frames=10)
    assert 1 <= len(frames) <= 10 and len(frames) == len(times)
    steps = np.diff(times)
    assert np.allclose(steps, steps[0]) and steps[0] >= 0.4 - 1e-6
    assert times[0] == 0.0 and times[-1] < 4.0


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_sample_frames_rotated_source_display_orientation(media_dir: Path, settings: Settings) -> None:
    probed, probed_times = sample_frames(media_dir / "rotated.mp4", 0.5, 1.5, fps=4.0, width=120,
                                         settings=settings, cancel=None)
    hinted, _ = sample_frames(media_dir / "rotated.mp4", 0.5, 1.5, fps=4.0, width=120, settings=settings,
                              cancel=None, display_size_hint=(240, 320))
    assert len(probed) == 4 and probed_times == pytest.approx([0.5, 0.75, 1.0, 1.25])
    assert probed[0].shape == hinted[0].shape == (160, 120, 3)
    assert np.array_equal(probed[1], hinted[1])


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_sample_frames_errors(media_dir: Path, settings: Settings) -> None:
    with pytest.raises(ValueError):
        sample_frames(media_dir / "bw.mp4", 2.0, 2.0, fps=4.0, width=64, settings=settings, cancel=None)
    token = CancelToken()
    token.cancel()
    with pytest.raises(CancelledError):
        sample_frames(media_dir / "bw.mp4", 0.0, 1.0, fps=4.0, width=64, settings=settings, cancel=token)
    with pytest.raises(MediaError):
        sample_frames(media_dir / "garbage.mp4", 0.0, 1.0, fps=4.0, width=64, settings=settings, cancel=None)
