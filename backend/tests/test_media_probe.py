"""Tests for clipper.media.probe: pure ffprobe-JSON normalisation + real ffprobe runs."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import requires_ffmpeg

from clipper.config import Settings
from clipper.errors import CancelledError, MediaError
from clipper.media.probe import (
    PROBE_VERSION,
    media_info_from_ffprobe,
    normalize_rotation,
    parse_rate,
    probe,
    select_video_stream,
    stream_fps,
    stream_rotation,
    tool_version,
)
from clipper.procs import CancelToken, run


# ----------------------------------------------------------------------------- helpers
def _video_stream(**over: Any) -> dict[str, Any]:
    s: dict[str, Any] = {
        "index": 0, "codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
        "pix_fmt": "yuv420p", "avg_frame_rate": "30000/1001", "r_frame_rate": "30000/1001",
        "duration": "12.000000", "disposition": {"attached_pic": 0},
    }
    s.update(over)
    return s


def _audio_stream(**over: Any) -> dict[str, Any]:
    s: dict[str, Any] = {
        "index": 1, "codec_type": "audio", "codec_name": "aac", "channels": 2, "sample_rate": "48000",
        "duration": "12.010000", "disposition": {"attached_pic": 0},
    }
    s.update(over)
    return s


def _data(streams: list[dict[str, Any]], **fmt_over: Any) -> dict[str, Any]:
    fmt: dict[str, Any] = {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "12.010000",
                           "start_time": "0.000000", "size": "123456"}
    fmt.update(fmt_over)
    return {"streams": streams, "format": fmt}


def _ffmpeg(settings: Settings, *args: str) -> None:
    run([settings.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y", *args], timeout=120)


def _make_av(settings: Settings, out: Path, *, duration: float = 2.0, size: str = "320x240",
             audio: bool = True, video: bool = True) -> Path:
    args: list[str] = []
    if video:
        args += ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25:duration={duration}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={duration}"]
    if video:
        args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-b:a", "64k"]
    _ffmpeg(settings, *args, str(out))
    return out


# ----------------------------------------------------------------------------- pure unit tests
def test_probe_version_is_int() -> None:
    assert isinstance(PROBE_VERSION, int) and PROBE_VERSION >= 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [("30000/1001", 30000 / 1001), ("25/1", 25.0), ("24", 24.0), ("0/0", None), ("1/0", None),
     ("", None), (None, None), ("abc", None), ("nan/1", None)],
)
def test_parse_rate(value: Any, expected: float | None) -> None:
    got = parse_rate(value)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


@pytest.mark.parametrize(
    ("degrees", "expected"),
    [(0, 0), (90, 90), (-90, 270), (180, 180), (-180, 180), (270, 270), (-270, 90), (360, 0),
     (450, 90), (89.6, 90), (float("nan"), 0)],
)
def test_normalize_rotation(degrees: float, expected: int) -> None:
    assert normalize_rotation(degrees) == expected


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        # Display matrix rotation is counter-clockwise: -90 (phone portrait) == 90 clockwise.
        ({"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]}, 90),
        ({"side_data_list": [{"side_data_type": "Display Matrix", "rotation": 90}]}, 270),
        ({"side_data_list": [{"side_data_type": "Display Matrix", "rotation": 180}]}, 180),
        ({"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -180}]}, 180),
        ({"side_data_list": [{"side_data_type": "Display Matrix", "rotation": 0}]}, 0),
        # Legacy tag is clockwise.
        ({"tags": {"rotate": "90"}}, 90),
        ({"tags": {"rotate": "270"}}, 270),
        # Side data wins over the tag; non-rotation side data is ignored.
        ({"side_data_list": [{"side_data_type": "CPB properties"}, {"rotation": -90}],
          "tags": {"rotate": "180"}}, 90),
        ({"side_data_list": [{"side_data_type": "CPB properties"}], "tags": {"rotate": "180"}}, 180),
        ({"tags": {"rotate": "garbage"}}, 0),
        ({}, 0),
    ],
)
def test_stream_rotation(stream: dict[str, Any], expected: int) -> None:
    assert stream_rotation(stream) == expected


def test_basic_media_info() -> None:
    info = media_info_from_ffprobe(_data([_video_stream(), _audio_stream()]), size_bytes=999)
    assert info.duration == pytest.approx(12.01)
    assert info.size_bytes == 999
    assert info.container.startswith("mov")
    assert info.has_video and info.has_audio
    assert (info.width, info.height, info.coded_width, info.coded_height) == (1920, 1080, 1920, 1080)
    assert info.rotation == 0
    assert info.fps == pytest.approx(29.97, abs=0.01)
    assert info.video_codec == "h264" and info.pix_fmt == "yuv420p"
    assert info.audio_codec == "aac" and info.audio_channels == 2 and info.audio_sample_rate == 48000
    assert info.start_time == 0.0


def test_size_falls_back_to_format_size() -> None:
    info = media_info_from_ffprobe(_data([_video_stream(), _audio_stream()]))
    assert info.size_bytes == 123456


@pytest.mark.parametrize(("ccw", "cw", "dims"), [(-90, 90, (1080, 1920)), (90, 270, (1080, 1920)),
                                                  (180, 180, (1920, 1080))])
def test_rotation_swaps_display_dimensions(ccw: int, cw: int, dims: tuple[int, int]) -> None:
    v = _video_stream(side_data_list=[{"side_data_type": "Display Matrix", "rotation": ccw}])
    info = media_info_from_ffprobe(_data([v, _audio_stream()]))
    assert info.rotation == cw
    assert (info.width, info.height) == dims
    assert (info.coded_width, info.coded_height) == (1920, 1080)


@pytest.mark.parametrize("sar, rotate, dims", [
    ("4:3", None, (1920, 1080)),          # HDV 1440x1080 anamorphic -> 16:9 display
    ("4:3", "90", (1080, 1920)),          # rotated: the pixel shape turns with the frame
    ("1:1", None, (1440, 1080)),
    ("0:1", None, (1440, 1080)),          # unknown pixel shape = square
    ("N/A", None, (1440, 1080)),
    (None, None, (1440, 1080)),
])
def test_display_size_applies_sample_aspect_ratio(sar: str | None, rotate: str | None,
                                                 dims: tuple[int, int]) -> None:
    # Review finding 16: width/height are documented as display size but ignored the SAR.
    over: dict[str, Any] = {"width": 1440, "height": 1080, "sample_aspect_ratio": sar}
    if rotate:
        over["tags"] = {"rotate": rotate}
    info = media_info_from_ffprobe(_data([_video_stream(**over)]))
    assert (info.width, info.height) == dims
    assert (info.coded_width, info.coded_height) == (1440, 1080)
    ntsc = media_info_from_ffprobe(_data([_video_stream(width=720, height=480, sample_aspect_ratio="8:9")]))
    assert (ntsc.width, ntsc.height) == (640, 480)


def test_attached_picture_is_not_video() -> None:
    cover = _video_stream(codec_name="mjpeg", width=600, height=600, disposition={"attached_pic": 1},
                          avg_frame_rate="0/0", r_frame_rate="90000/1")
    info = media_info_from_ffprobe(_data([_audio_stream(index=0), {**cover, "index": 1}]))
    assert not info.has_video and info.has_audio
    assert info.width == 0 and info.video_codec is None and info.fps == 0.0


def test_attached_picture_skipped_when_real_video_follows() -> None:
    cover = _video_stream(index=0, codec_name="png", width=64, height=64, disposition={"attached_pic": 1})
    real = _video_stream(index=1, width=640, height=360)
    assert select_video_stream([cover, real]) is real
    info = media_info_from_ffprobe(_data([cover, real, _audio_stream(index=2)]))
    assert (info.width, info.height, info.video_codec) == (640, 360, "h264")


def test_fps_fallbacks() -> None:
    assert stream_fps({"avg_frame_rate": "0/0", "r_frame_rate": "25/1"}) == pytest.approx(25.0)
    assert stream_fps({"avg_frame_rate": "24000/1001"}) == pytest.approx(23.976, abs=1e-3)
    assert stream_fps({"avg_frame_rate": "0/0", "r_frame_rate": "0/0"}) == 0.0
    assert stream_fps({}) == 0.0


def test_duration_falls_back_to_longest_stream() -> None:
    data = _data([_video_stream(duration="9.5"), _audio_stream(duration="10.25")])
    del data["format"]["duration"]
    assert media_info_from_ffprobe(data).duration == pytest.approx(10.25)


def test_duration_from_matroska_tag() -> None:
    v = _video_stream(tags={"DURATION": "00:15:55.123000000"})
    del v["duration"]
    a = _audio_stream(tags={"DURATION": "00:15:54.000000000"})
    del a["duration"]
    data = _data([v, a], format_name="matroska,webm")
    data["format"]["duration"] = "N/A"
    assert media_info_from_ffprobe(data).duration == pytest.approx(955.123)


def test_audio_only() -> None:
    info = media_info_from_ffprobe(_data([_audio_stream(index=0)], format_name="mp3"))
    assert info.has_audio and not info.has_video
    assert info.width == 0 and info.fps == 0.0 and info.rotation == 0


def test_video_only() -> None:
    info = media_info_from_ffprobe(_data([_video_stream()]))
    assert info.has_video and not info.has_audio
    assert info.audio_codec is None and info.audio_channels is None


def test_start_time_reported() -> None:
    info = media_info_from_ffprobe(_data([_video_stream(), _audio_stream()], start_time="1.400000"))
    assert info.start_time == pytest.approx(1.4)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"streams": [], "format": {"duration": "10"}},
        # what random bytes look like to ffprobe: a "subtitle" stream, no duration
        {"streams": [{"codec_type": "subtitle", "codec_name": "text"}], "format": {"format_name": "lrc"}},
        _data([_video_stream(duration="0")], duration="0"),
        _data([_video_stream(duration="N/A")], duration="N/A"),
        _data([_audio_stream(duration="-1")], duration="-3"),
        # only cover art, no audio
        _data([_video_stream(disposition={"attached_pic": 1})]),
    ],
)
def test_unreadable_inputs_raise_media_error(data: dict[str, Any]) -> None:
    with pytest.raises(MediaError) as exc:
        media_info_from_ffprobe(data)
    assert "not a readable video" in exc.value.user_message


def test_missing_file_raises_media_error(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(MediaError) as exc:
        probe(tmp_path / "nope.mp4", settings=settings)
    assert str(tmp_path) not in exc.value.user_message


def test_tool_version_missing_binary() -> None:
    assert tool_version("definitely-not-a-real-binary-xyz") is None


# ----------------------------------------------------------------------------- real ffprobe
@pytest.mark.ffmpeg
@requires_ffmpeg
def test_probe_synthetic_mp4(settings: Settings, tmp_path: Path) -> None:
    src = _make_av(settings, tmp_path / "in.mp4", duration=2.0)
    info = probe(src, settings=settings)
    assert info.has_video and info.has_audio
    assert (info.width, info.height) == (320, 240)
    assert (info.coded_width, info.coded_height) == (320, 240)
    assert info.rotation == 0
    assert info.fps == pytest.approx(25.0)
    assert info.duration == pytest.approx(2.0, abs=0.1)
    assert info.size_bytes == src.stat().st_size
    assert info.video_codec == "h264" and info.audio_codec == "aac"
    assert info.audio_sample_rate == 44100 and info.audio_channels == 1
    assert info.pix_fmt == "yuv420p"
    assert "mp4" in info.container


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_probe_audio_only_m4a(settings: Settings, tmp_path: Path) -> None:
    src = _make_av(settings, tmp_path / "a.m4a", duration=1.5, video=False)
    info = probe(src, settings=settings)
    assert info.has_audio and not info.has_video
    assert info.width == 0 and info.height == 0 and info.fps == 0.0
    assert info.duration == pytest.approx(1.5, abs=0.1)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_probe_m4a_with_cover_art_is_audio_only(settings: Settings, tmp_path: Path) -> None:
    audio = _make_av(settings, tmp_path / "a.m4a", duration=1.0, video=False)
    cover = tmp_path / "cover.png"
    _ffmpeg(settings, "-f", "lavfi", "-i", "color=red:size=64x64:duration=1", "-frames:v", "1", str(cover))
    out = tmp_path / "cover.m4a"
    _ffmpeg(settings, "-i", str(audio), "-i", str(cover), "-map", "0", "-map", "1", "-c", "copy",
            "-disposition:v:0", "attached_pic", str(out))
    info = probe(out, settings=settings)
    assert info.has_audio and not info.has_video


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_probe_video_without_audio(settings: Settings, tmp_path: Path) -> None:
    src = _make_av(settings, tmp_path / "v.mp4", duration=1.0, audio=False)
    info = probe(src, settings=settings)
    assert info.has_video and not info.has_audio
    assert info.audio_codec is None


@pytest.mark.ffmpeg
@requires_ffmpeg
@pytest.mark.parametrize("fmt, ext", [("webm", "webm"), ("matroska", "mkv")])
def test_probe_live_muxed_file_without_duration_metadata(settings: Settings, tmp_path: Path, fmt: str,
                                                         ext: str) -> None:
    # Review finding 15: MediaRecorder-style (live) WebM/MKV carry no duration anywhere and were
    # rejected as "not a readable video" although they decode completely.
    src = tmp_path / f"live.{ext}"
    _ffmpeg(settings, "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=15", "-f", "lavfi",
            "-i", "sine=frequency=440:duration=15", "-c:v", "libvpx", "-b:v", "200k", "-deadline", "realtime",
            "-cpu-used", "8", "-c:a", "libopus", "-f", fmt, "-live", "1", str(src))
    raw = run([settings.ffprobe, "-v", "error", "-show_entries", "format=duration:stream=duration", "-of", "json",
               str(src)], timeout=60).stdout.decode()
    assert '"duration"' not in raw                       # the fixture really has no duration metadata
    info = probe(src, settings=settings)
    assert info.has_video and info.has_audio
    assert info.duration == pytest.approx(15.0, abs=0.1)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_probe_anamorphic_source_reports_display_size(settings: Settings, tmp_path: Path) -> None:
    src = tmp_path / "hdv.mp4"
    _ffmpeg(settings, "-f", "lavfi", "-i", "testsrc2=size=1440x1080:rate=25:duration=1", "-vf", "setsar=4/3",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src))
    info = probe(src, settings=settings)
    assert (info.width, info.height) == (1920, 1080) and (info.coded_width, info.coded_height) == (1440, 1080)


def _decode_first_frame(settings: Settings, src: Path, w: int, h: int, *, autorotate: bool) -> np.ndarray:
    args = [settings.ffmpeg, "-hide_banner", "-nostdin", "-v", "error"]
    if not autorotate:
        args.append("-noautorotate")
    args += ["-i", str(src), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = run(args, timeout=60).stdout
    assert len(raw) == w * h
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w).astype(np.int16)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_probe_rotated_video_matches_ffmpeg_autorotate(settings: Settings, tmp_path: Path) -> None:
    src = _make_av(settings, tmp_path / "in.mp4", duration=1.0, size="320x240")
    rotated = tmp_path / "rot.mp4"
    # -display_rotation is counter-clockwise; it is an *input* option applied to a stream copy.
    _ffmpeg(settings, "-display_rotation", "90", "-i", str(src), "-c", "copy", str(rotated))
    info = probe(rotated, settings=settings)
    assert info.rotation == 270  # 90° counter-clockwise == 270° clockwise
    assert (info.width, info.height) == (240, 320)
    assert (info.coded_width, info.coded_height) == (320, 240)

    # The clockwise rotation we report must reproduce what ffmpeg's autorotate shows.
    stored = _decode_first_frame(settings, rotated, 320, 240, autorotate=False)
    shown = _decode_first_frame(settings, rotated, info.width, info.height, autorotate=True)
    ours = np.rot90(stored, k=-info.rotation // 90)  # np.rot90: positive k = counter-clockwise
    assert ours.shape == shown.shape
    assert float(np.abs(ours - shown).mean()) < 3.0
    wrong = np.rot90(stored, k=info.rotation // 90)
    assert float(np.abs(wrong - shown).mean()) > 10.0


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_probe_random_bytes_is_media_error(settings: Settings, tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(random.Random(1234).randbytes(50_000))
    with pytest.raises(MediaError) as exc:
        probe(junk, settings=settings)
    assert "not a readable video" in exc.value.user_message
    assert str(tmp_path) not in exc.value.user_message


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_probe_truncated_mp4_is_media_error(settings: Settings, tmp_path: Path) -> None:
    src = _make_av(settings, tmp_path / "in.mp4", duration=1.0)
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(src.read_bytes()[:600])
    with pytest.raises(MediaError) as exc:
        probe(broken, settings=settings)
    assert "not a readable video" in exc.value.user_message


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_probe_respects_cancel(settings: Settings, tmp_path: Path) -> None:
    src = _make_av(settings, tmp_path / "in.mp4", duration=1.0)
    token = CancelToken()
    token.cancel()
    with pytest.raises(CancelledError):
        probe(src, settings=settings, cancel=token)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_tool_version_real(settings: Settings) -> None:
    ffmpeg_v = tool_version(settings.ffmpeg)
    ffprobe_v = tool_version(settings.ffprobe)
    assert ffmpeg_v is not None and ffmpeg_v.startswith("ffmpeg version")
    assert ffprobe_v is not None and ffprobe_v.startswith("ffprobe version")
    assert "\n" not in ffmpeg_v
