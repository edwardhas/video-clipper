"""Media inspection with ffprobe.

``probe`` runs ``ffprobe -show_format -show_streams -of json`` (argument list, never a
shell string) and normalises the result into :class:`clipper.models.MediaInfo`:

* the video stream is the first ``codec_type == "video"`` stream that is not an
  attached picture (cover art in m4a/mp3/mkv files is ignored);
* the audio stream is the first audio stream (the same one ``-map 0:a:0`` selects);
* ``rotation`` is degrees *clockwise* needed to display the stored frames, taken from
  the display-matrix side data (whose ``rotation`` value is counter-clockwise, so
  ``-90`` means 90° clockwise) or the legacy ``rotate`` tag (already clockwise);
  display ``width``/``height`` are swapped for 90/270;
* display ``width`` applies a non-square ``sample_aspect_ratio`` (anamorphic DVD/HDV
  sources: 1440x1080 at SAR 4:3 is displayed 1920x1080); ``coded_*`` stay storage pixels;
* ``fps`` comes from ``avg_frame_rate`` (fallback ``r_frame_rate``; ``0/0`` ignored);
* ``duration`` comes from ``format.duration``, else the longest stream duration; for
  live-muxed WebM/MKV (browser MediaRecorder output) that carry neither, :func:`probe`
  measures it from the last packet timestamps.

Unreadable input (ffprobe failure, no audio/video streams, no positive duration)
raises :class:`MediaError` with a plain-language message.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..config import Settings
from ..errors import ClipperError, MediaError
from ..models import MediaInfo
from ..procs import CancelToken, run

PROBE_VERSION = 2

UNREADABLE_MESSAGE = (
    "This file is not a readable video. It may be damaged, incomplete, or in an unsupported format."
)
_PROBE_TIMEOUT_S = 120.0
_SCAN_TIMEOUT_S = 900.0      # packet scan (demux only) of a file without duration metadata
_MAX_SANE_FPS = 1000.0


def probe(path: Path, *, settings: Settings, cancel: CancelToken | None = None) -> MediaInfo:
    """Inspect ``path`` with ffprobe and return normalised :class:`MediaInfo`."""
    path = Path(path)
    if not path.is_file():
        raise MediaError(UNREADABLE_MESSAGE, detail="probe: input is not a regular file")
    data = _run_ffprobe(path, settings=settings, cancel=cancel)
    _fill_missing_duration(path, data, settings=settings, cancel=cancel)
    return media_info_from_ffprobe(data, size_bytes=path.stat().st_size)


def _fill_missing_duration(path: Path, data: dict[str, Any], *, settings: Settings,
                           cancel: CancelToken | None) -> None:
    """Live-muxed WebM/MKV (e.g. browser MediaRecorder) have no duration in the header or
    stream tags although they decode completely: measure it from the packet timestamps.

    Only runs when there is an audio/video stream and no usable duration; random bytes
    produce no packets and stay unreadable.
    """
    fmt = data.get("format") if isinstance(data.get("format"), dict) else {}
    streams = [s for s in data.get("streams") or [] if isinstance(s, dict)]
    av = [s for s in (select_video_stream(streams),
                      next((s for s in streams if s.get("codec_type") == "audio"), None)) if s is not None]
    if not av:
        return
    known = _duration(fmt, av)
    if known is not None and known > 0:
        return
    end: float | None = None

    def on_line(line: str) -> None:
        nonlocal end
        fields = line.strip().split(",")
        pts = _to_float(fields[0]) if fields else None
        if pts is None:
            return
        dur = _to_float(fields[1]) if len(fields) > 1 else None
        last = pts + (dur if dur is not None and dur > 0 else 0.0)
        if end is None or last > end:
            end = last

    run([settings.ffprobe, "-hide_banner", "-v", "error", "-show_entries", "packet=pts_time,duration_time",
         "-of", "csv=p=0", "-i", str(path)],
        cancel=cancel, timeout=_SCAN_TIMEOUT_S, stdout_line_cb=on_line, error_message=UNREADABLE_MESSAGE)
    if end is None:
        return
    duration = end - (_to_float(fmt.get("start_time")) or 0.0)
    if duration > 0:
        new_fmt = dict(fmt)
        new_fmt["duration"] = f"{duration:.6f}"
        data["format"] = new_fmt


def _run_ffprobe(path: Path, *, settings: Settings, cancel: CancelToken | None) -> dict[str, Any]:
    args = [
        settings.ffprobe, "-hide_banner", "-v", "error",
        "-show_format", "-show_streams", "-of", "json",
        "-i", str(path),
    ]
    result = run(args, cancel=cancel, timeout=_PROBE_TIMEOUT_S, error_message=UNREADABLE_MESSAGE)
    try:
        data = json.loads(result.stdout.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError(UNREADABLE_MESSAGE, detail="ffprobe returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise MediaError(UNREADABLE_MESSAGE, detail="ffprobe returned unexpected JSON")
    return data


def media_info_from_ffprobe(data: dict[str, Any], *, size_bytes: int | None = None) -> MediaInfo:
    """Build :class:`MediaInfo` from parsed ``ffprobe -of json`` output (pure; unit-testable)."""
    fmt = data.get("format") if isinstance(data.get("format"), dict) else {}
    streams = [s for s in data.get("streams") or [] if isinstance(s, dict)]
    video = select_video_stream(streams)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None and audio is None:
        raise MediaError(UNREADABLE_MESSAGE, detail="probe: no audio or video streams")

    duration = _duration(fmt, [s for s in (video, audio) if s is not None] or streams)
    if duration is None or duration <= 0:
        raise MediaError(UNREADABLE_MESSAGE, detail="probe: missing or non-positive duration")

    if size_bytes is None:
        size_bytes = _to_int(fmt.get("size")) or 0

    info: dict[str, Any] = {
        "duration": duration,
        "size_bytes": int(size_bytes),
        "container": str(fmt.get("format_name") or "unknown"),
        "has_video": video is not None,
        "has_audio": audio is not None,
        "start_time": _to_float(fmt.get("start_time")) or 0.0,
    }
    if video is not None:
        info.update(_video_fields(video))
    if audio is not None:
        info.update(
            audio_codec=audio.get("codec_name"),
            audio_channels=_to_int(audio.get("channels")),
            audio_sample_rate=_to_int(audio.get("sample_rate")),
        )
    return MediaInfo(**info)


def select_video_stream(streams: list[dict[str, Any]]) -> dict[str, Any] | None:
    """First real video stream, skipping attached pictures (cover art)."""
    for s in streams:
        if s.get("codec_type") != "video":
            continue
        disposition = s.get("disposition") if isinstance(s.get("disposition"), dict) else {}
        if _to_int(disposition.get("attached_pic")) == 1:
            continue
        return s
    return None


def _video_fields(video: dict[str, Any]) -> dict[str, Any]:
    coded_w = _to_int(video.get("width")) or 0
    coded_h = _to_int(video.get("height")) or 0
    rotation = stream_rotation(video)
    width, height = display_dimensions(coded_w, coded_h, sample_aspect_ratio(video), rotation)
    return {
        "width": width,
        "height": height,
        "coded_width": coded_w,
        "coded_height": coded_h,
        "rotation": rotation,
        "fps": stream_fps(video),
        "video_codec": video.get("codec_name"),
        "pix_fmt": video.get("pix_fmt"),
    }


def sample_aspect_ratio(stream: dict[str, Any]) -> tuple[int, int] | None:
    """Pixel shape ``(num, den)`` from ``sample_aspect_ratio``; None when square/unknown (``0:1``, N/A)."""
    text = str(stream.get("sample_aspect_ratio") or "")
    num_s, sep, den_s = text.partition(":")
    try:
        num, den = int(num_s), int(den_s)
    except ValueError:
        return None
    if not sep or num <= 0 or den <= 0 or num == den or not 0.25 <= num / den <= 4.0:   # square, unknown or bogus
        return None
    return num, den


def display_dimensions(coded_w: int, coded_h: int, sar: tuple[int, int] | None, rotation: int) -> tuple[int, int]:
    """Display ``(width, height)``: non-square pixels widen/narrow the width (even), then 90/270 swap.

    FFmpeg's auto-rotation inverts the pixel shape together with the frame, so scaling the
    rotated frame to this size is correct for every orientation.
    """
    width, height = coded_w, coded_h
    if sar is not None and coded_w > 0:
        width = max(2, int(round(coded_w * sar[0] / sar[1] / 2.0)) * 2)
    return (height, width) if rotation in (90, 270) else (width, height)


def stream_rotation(stream: dict[str, Any]) -> int:
    """Clockwise degrees (0/90/180/270) needed to display the stream upright.

    The display matrix ``rotation`` reported by ffprobe is counter-clockwise (``-90`` is
    what phones write for portrait video, i.e. 90° clockwise). The legacy ``rotate``
    tag is already clockwise. Side data wins over the tag when both exist.
    """
    for entry in stream.get("side_data_list") or []:
        if not isinstance(entry, dict) or "rotation" not in entry:
            continue
        ccw = _to_float(entry.get("rotation"))
        if ccw is not None:
            return normalize_rotation(-ccw)
    tags = stream.get("tags") if isinstance(stream.get("tags"), dict) else {}
    cw = _to_float(tags.get("rotate"))
    if cw is not None:
        return normalize_rotation(cw)
    return 0


def normalize_rotation(degrees: float) -> int:
    """Snap any angle to the nearest of 0/90/180/270 (clockwise, modulo 360)."""
    if not math.isfinite(degrees):
        return 0
    return int(round(degrees / 90.0)) % 4 * 90


def stream_fps(stream: dict[str, Any]) -> float:
    """Average frame rate; falls back to ``r_frame_rate``; 0.0 when unknown."""
    for key in ("avg_frame_rate", "r_frame_rate"):
        fps = parse_rate(stream.get(key))
        if fps is not None and 0 < fps <= _MAX_SANE_FPS:
            return fps
    return 0.0


def parse_rate(value: Any) -> float | None:
    """Parse ``"30000/1001"`` / ``"25"``; ``None`` for ``0/0`` or garbage."""
    if value is None:
        return None
    text = str(value).strip()
    num_s, sep, den_s = text.partition("/")
    try:
        num = float(num_s)
        den = float(den_s) if sep else 1.0
    except ValueError:
        return None
    if den == 0 or not math.isfinite(num) or not math.isfinite(den):
        return None
    rate = num / den
    return rate if math.isfinite(rate) else None


def _duration(fmt: dict[str, Any], streams: list[dict[str, Any]]) -> float | None:
    value = _to_float(fmt.get("duration"))
    if value is not None and value > 0:
        return value
    candidates = [d for d in (_stream_duration(s) for s in streams) if d is not None and d > 0]
    return max(candidates) if candidates else None


def _stream_duration(stream: dict[str, Any]) -> float | None:
    value = _to_float(stream.get("duration"))
    if value is not None:
        return value
    tags = stream.get("tags") if isinstance(stream.get("tags"), dict) else {}
    for key in ("DURATION", "duration"):  # Matroska/WebM store it as a "HH:MM:SS.nnnnnnnnn" tag
        parsed = _parse_clock(tags.get(key))
        if parsed is not None:
            return parsed
    return None


def _parse_clock(value: Any) -> float | None:
    if not isinstance(value, str) or ":" not in value:
        return None
    parts = value.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    except ValueError:
        return None
    total = hours * 3600 + minutes * 60 + seconds
    return total if math.isfinite(total) else None


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _to_int(value: Any) -> int | None:
    as_float = _to_float(value)
    return int(as_float) if as_float is not None else None


def tool_version(binary: str) -> str | None:
    """First line of ``<binary> -version`` (e.g. ``ffmpeg version 9.0.1 …``), or None."""
    try:
        result = run([binary, "-version"], timeout=20, error_message="version check failed")
    except (ClipperError, OSError, ValueError):  # includes MissingDependencyError
        return None
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        if line.strip():
            return line.strip()
    return None
