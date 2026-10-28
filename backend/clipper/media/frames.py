"""Still frames and raw frame sampling with FFmpeg.

* :func:`extract_frame` / :func:`extract_frames` write JPEG stills (review frames, UI
  framing previews). Seeking is accurate (``-ss`` before ``-i`` while re-decoding).
* :func:`sample_frames` streams ``bgr24`` rawvideo from FFmpeg into NumPy arrays for
  local analysis (face framing). The output size is forced to an explicit even
  ``W x H`` so the byte stride per frame is exact. It seeks fast (``-noaccurate_seek``)
  and resamples with ``fps=<rate>:start_time=0``, so sample ``i`` is the frame *on
  screen* at ``src_start + i / rate`` even for variable-frame-rate sources that hold
  one frame for seconds.

All times passed in are **source seconds** (``src_*``). FFmpeg auto-rotates decoded
frames, and non-square pixels (anamorphic sources) are scaled square, so every size
reported or produced here is in display orientation and proportions. HDR (HLG/PQ)
sources are tone-mapped to BT.709 first (:mod:`clipper.media.color`).
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import Settings
from ..errors import MediaError
from ..procs import CancelToken, run
from ..storage import atomic_path
from .color import sdr_filter_for
from .probe import display_dimensions, sample_aspect_ratio

_FRAME_ERROR = "Could not read a video frame from the source."
# Square pixels: non-square (anamorphic) frames are widened/narrowed to their display width.
# Same rule as probe.sample_aspect_ratio (so stills match MediaInfo and the render): a pixel
# aspect outside 1/4..4 is bogus and ignored. The range is symmetric, so it also holds after
# auto-rotation inverts the aspect.
_SQUARE_PIXELS = "scale=w='max(2,trunc(iw*if(between(sar,0.25,4),sar,1)/2+0.5)*2)':h=ih,setsar=1"


def _even_floor(value: float) -> int:
    return max(2, int(value) // 2 * 2)


def _even_round(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def scaled_size(display_w: int, display_h: int, width: int) -> tuple[int, int]:
    """Even output size for ``width`` (never upscaled) keeping the display aspect."""
    if display_w <= 0 or display_h <= 0:
        raise ValueError("display size must be positive")
    out_w = _even_floor(min(width, display_w))
    out_h = _even_round(out_w * display_h / display_w)
    return out_w, out_h


def rotation_from_stream(stream: dict[str, Any]) -> int:
    """Clockwise display rotation (0/90/180/270) from an ffprobe stream dict."""
    raw: float | None = None
    for item in stream.get("side_data_list") or []:
        if isinstance(item, dict) and "rotation" in item:
            try:
                raw = float(item["rotation"])
            except (TypeError, ValueError):
                raw = None
            break
    if raw is None:
        tag = (stream.get("tags") or {}).get("rotate")
        try:
            raw = float(tag) if tag is not None else 0.0
        except (TypeError, ValueError):
            raw = 0.0
    return int(round(raw / 90.0)) * 90 % 360


def display_size(src: Path, *, settings: Settings, cancel: CancelToken | None = None) -> tuple[int, int]:
    """Display width/height of the first video stream (rotation applied)."""
    result = run(
        [settings.ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(src)],
        cancel=cancel, timeout=60, error_message="This file is not a readable video.",
    )
    try:
        streams = json.loads(result.stdout.decode("utf-8", "replace") or "{}").get("streams") or []
    except json.JSONDecodeError as exc:
        raise MediaError("This file is not a readable video.") from exc
    if not streams:
        raise MediaError("This file has no video picture to analyse.")
    stream = streams[0]
    w, h = int(stream.get("width") or 0), int(stream.get("height") or 0)
    if w <= 0 or h <= 0:
        raise MediaError("This file has no video picture to analyse.")
    return display_dimensions(w, h, sample_aspect_ratio(stream), rotation_from_stream(stream))


# ----------------------------------------------------------------------------- JPEG stills
def extract_frame(src: Path, t: float, out_jpg: Path, *, max_width: int = 640, settings: Settings,
                  cancel: CancelToken | None = None) -> Path:
    """Write the frame at source time ``t`` as a JPEG no wider than ``max_width`` (even dims).

    The file is written atomically; on failure ``out_jpg`` is left untouched.
    """
    if max_width < 2:
        raise ValueError("max_width must be at least 2")
    seek = max(0.0, float(t))
    sdr = sdr_filter_for(Path(src), settings=settings, cancel=cancel)
    vf = (f"{sdr + ',' if sdr else ''}{_SQUARE_PIXELS},"
          f"scale=w='trunc(min(iw,{int(max_width)})/2)*2':h=-2:flags=area")
    out_jpg = Path(out_jpg)
    with atomic_path(out_jpg) as tmp:
        run(
            [settings.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y",
             "-ss", f"{seek:.3f}", "-i", str(src),
             "-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", "1", "-vf", vf,
             "-q:v", "3", "-f", "image2", "-update", "1", str(tmp)],
            cancel=cancel, timeout=120, error_message=_FRAME_ERROR,
        )
        if tmp.stat().st_size == 0:
            raise MediaError("No video frame exists at the requested time.")
    return out_jpg


def _frame_name(index: int, t: float, prefix: str | None) -> str:
    if prefix:
        return f"{prefix}_{index}.jpg"
    return f"t{int(round(max(0.0, t) * 1000)):09d}.jpg"


def extract_frames(src: Path, times: Sequence[float], out_dir: Path, *, max_width: int = 512,
                   settings: Settings, cancel: CancelToken | None = None,
                   prefix: str | None = None) -> list[Path]:
    """Extract one JPEG per source time into ``out_dir``; returns paths in input order.

    Files are named ``<prefix>_<n>.jpg`` when ``prefix`` is given (e.g. a candidate id),
    otherwise ``t<milliseconds>.jpg`` so different callers sharing a directory never collide.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, t in enumerate(times):
        if cancel is not None:
            cancel.raise_if_cancelled()
        paths.append(extract_frame(src, t, out_dir / _frame_name(index, t, prefix), max_width=max_width,
                                   settings=settings, cancel=cancel))
    return paths


# ----------------------------------------------------------------------------- raw sampling
class FrameCollector:
    """Splits a raw ``bgr24`` byte stream into ``(h, w, 3)`` arrays, keeping at most ``limit``."""

    def __init__(self, width: int, height: int, limit: int) -> None:
        self.width = width
        self.height = height
        self.limit = limit
        self.frame_bytes = width * height * 3
        self.frames: list[np.ndarray] = []
        self.dropped = 0
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> None:
        self._buf.extend(chunk)
        fb = self.frame_bytes
        while len(self._buf) >= fb:
            if len(self.frames) < self.limit:
                arr = np.frombuffer(bytes(self._buf[:fb]), dtype=np.uint8)
                self.frames.append(arr.reshape(self.height, self.width, 3).copy())
            else:
                self.dropped += 1
            del self._buf[:fb]

    @property
    def leftover(self) -> int:
        return len(self._buf)


def effective_fps(fps: float, duration: float, max_frames: int) -> Fraction:
    """Sampling rate as an exact fraction, lowered so ``duration`` yields ≤ ``max_frames``."""
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("fps must be positive")
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    rate = min(float(fps), max_frames / duration) if duration > 0 else float(fps)
    frac = Fraction(rate).limit_denominator(1001)
    if frac <= 0:
        frac = Fraction(1, 1001)
    # limit_denominator may round up slightly; never exceed the frame budget.
    while duration > 0 and math.ceil(duration * frac) > max_frames and frac.numerator > 1:
        frac = Fraction(frac.numerator - 1, frac.denominator)
    return frac


def sample_frames(src: Path, src_start: float, src_end: float, *, fps: float, width: int, settings: Settings,
                  cancel: CancelToken | None, max_frames: int = 400,
                  display_size_hint: tuple[int, int] | None = None) -> tuple[list[np.ndarray], list[float]]:
    """Decode frames of ``[src_start, src_end)`` at ``fps`` (lowered to respect ``max_frames``).

    Returns BGR ``uint8`` arrays of shape ``(H, W, 3)`` (``W`` = ``width`` or the source width
    if smaller, both even, display orientation) and each frame's **source** time
    ``src_start + i / fps``. ``display_size_hint`` (e.g. ``(media.width, media.height)``)
    skips the ffprobe call; the explicit output size keeps the byte stride exact either way.
    """
    duration = float(src_end) - float(src_start)
    if duration <= 0:
        raise ValueError("src_end must be after src_start")
    if width < 2:
        raise ValueError("width must be at least 2")
    if display_size_hint and display_size_hint[0] > 0 and display_size_hint[1] > 0:
        disp_w, disp_h = display_size_hint
    else:
        disp_w, disp_h = display_size(src, settings=settings, cancel=cancel)
    out_w, out_h = scaled_size(disp_w, disp_h, width)
    rate = effective_fps(fps, duration, max_frames)
    collector = FrameCollector(out_w, out_h, max_frames)
    sdr = sdr_filter_for(Path(src), settings=settings, cancel=cancel)
    # Fast seek keeps the frame on screen at src_start; fps=..:start_time=0 then emits frame i at
    # exactly src_start + i/rate (an accurate seek would drop a held VFR frame and start late).
    vf = (f"fps={rate.numerator}/{rate.denominator}:start_time=0,{sdr + ',' if sdr else ''}"
          f"scale={out_w}:{out_h}:flags=area,setsar=1")
    run(
        [settings.ffmpeg, "-hide_banner", "-nostdin", "-v", "error",
         "-noaccurate_seek", "-ss", f"{max(0.0, float(src_start)):.3f}", "-i", str(src), "-t", f"{duration:.3f}",
         "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", vf, "-fps_mode", "passthrough",
         "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
        cancel=cancel, timeout=120 + duration * 3, stdout_bytes_cb=collector.feed,
        error_message=_FRAME_ERROR,
    )
    step = rate.denominator / rate.numerator
    frames: list[np.ndarray] = []
    times: list[float] = []
    for i, frame in enumerate(collector.frames):
        src_t = float(src_start) + i * step
        if src_t >= float(src_end) - 1e-6:
            break
        frames.append(frame)
        times.append(round(src_t, 6))
    return frames, times
