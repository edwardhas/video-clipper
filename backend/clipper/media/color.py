"""HDR detection and HDR -> SDR (BT.709) conversion filters.

Rendered clips, thumbnails and analysis frames are 8-bit SDR. An HDR source (HLG
``arib-std-b67`` or PQ ``smpte2084``, e.g. iPhone footage) must be tone-mapped to BT.709
*before* anything else happens to its pixels: otherwise the 8-bit H.264 output keeps the
BT.2020/HLG tags (HDR players show it at peak brightness, others washed out) and the sRGB
caption PNGs are composited into an HDR signal.

The conversion filter depends on the FFmpeg build, chosen once per binary by a capability
check:

* ``zscale`` (libzimg) + ``tonemap`` — Debian/Docker FFmpeg 7.1 and most distro builds:
  linearise, convert primaries, Hable tone map, back to BT.709 limited range;
* swscale colour management (FFmpeg >= 8, ``scale`` has ``out_transfer``) — e.g. Homebrew
  builds without libzimg;
* neither: no conversion (logged); the clip keeps the old behaviour.

Both chains read the frame's colour tags. Missing (or unsupported) primaries/matrix/range
tags of an HDR stream are filled with BT.2020 limited-range defaults first (``setparams``);
otherwise zscale fails ("no path between colorspaces") and swscale guesses BT.709.

SDR sources are never touched by these filters.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..config import Settings
from ..errors import CancelledError, ClipperError, InterruptedError_
from ..procs import CancelToken, run
from .probe import select_video_stream

log = logging.getLogger(__name__)

HDR_TRANSFERS = frozenset({"arib-std-b67", "smpte2084"})

ZSCALE_SDR = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
              "zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
SWSCALE_SDR = "scale=out_primaries=bt709:out_transfer=bt709:out_color_matrix=bt709:out_range=tv,format=yuv420p"
# Input tags the conversion chains handle; anything else on an HDR stream falls back to BT.2020.
_HDR_PRIMARIES = frozenset({"bt2020", "bt709", "smpte431", "smpte432"})
_HDR_MATRICES = frozenset({"bt2020nc", "bt2020c", "bt709"})


@dataclass(frozen=True)
class VideoColor:
    transfer: str | None = None
    primaries: str | None = None
    space: str | None = None
    range: str | None = None

    @property
    def is_hdr(self) -> bool:
        return self.transfer in HDR_TRANSFERS

    def hdr_input_tags(self) -> str:
        """``setparams`` giving an HDR stream complete colour tags (BT.2020 limited-range defaults)."""
        primaries = self.primaries if self.primaries in _HDR_PRIMARIES else "bt2020"
        matrix = self.space if self.space in _HDR_MATRICES else "bt2020nc"
        rng = self.range if self.range in ("tv", "pc") else "tv"
        return f"setparams=color_primaries={primaries}:color_trc={self.transfer}:colorspace={matrix}:range={rng}"


def _filter_help(ffmpeg: str, name: str) -> str:
    try:
        res = run([ffmpeg, "-hide_banner", "-h", f"filter={name}"], timeout=30, check=False,
                  error_message="filter check failed")
    except (ClipperError, OSError, ValueError):
        return ""
    text = (res.stdout or b"").decode("utf-8", "replace")
    return "" if "Unknown filter" in text else text


@lru_cache(maxsize=8)
def sdr_conversion(ffmpeg: str) -> str | None:
    """Filter chain converting HDR frames to 8-bit BT.709 limited-range ``yuv420p``; None if unsupported."""
    if _filter_help(ffmpeg, "zscale") and _filter_help(ffmpeg, "tonemap"):
        return ZSCALE_SDR
    if "out_transfer" in _filter_help(ffmpeg, "scale"):
        return SWSCALE_SDR
    log.warning("this FFmpeg build has neither zscale+tonemap nor swscale colour conversion; "
                "HDR sources are rendered without tone mapping")
    return None


_COLOR_CACHE: dict[tuple[str, str, int, int], VideoColor] = {}
_COLOR_CACHE_MAX = 64


def _read_video_color(ffprobe: str, path: str, cancel: CancelToken | None) -> VideoColor:
    res = run([ffprobe, "-v", "error", "-show_streams", "-of", "json", path], cancel=cancel, timeout=60,
              error_message="This file is not a readable video.")
    try:
        streams = json.loads(res.stdout.decode("utf-8", "replace") or "{}").get("streams") or []
    except (json.JSONDecodeError, AttributeError):
        streams = []
    video = select_video_stream([s for s in streams if isinstance(s, dict)])
    if video is None:
        return VideoColor()

    def tag(key: str) -> str | None:
        value = video.get(key)
        return str(value) if value not in (None, "", "unknown", "N/A") else None

    return VideoColor(transfer=tag("color_transfer"), primaries=tag("color_primaries"),
                      space=tag("color_space"), range=tag("color_range"))


def video_color(src: Path, *, settings: Settings, cancel: CancelToken | None = None) -> VideoColor:
    """Colour tags of the first real video stream (cached per file version).

    A missing or unreadable file reports no tags (treated as SDR) so the caller's own FFmpeg
    run produces its usual error; cancellation still propagates.
    """
    if cancel is not None:
        cancel.raise_if_cancelled()
    try:
        st = os.stat(src)
    except OSError:
        return VideoColor()
    key = (settings.ffprobe, str(src), st.st_size, st.st_mtime_ns)
    cached = _COLOR_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        color = _read_video_color(settings.ffprobe, str(src), cancel)
    except (CancelledError, InterruptedError_):
        raise
    except ClipperError:
        return VideoColor()
    if len(_COLOR_CACHE) >= _COLOR_CACHE_MAX:
        _COLOR_CACHE.clear()
    _COLOR_CACHE[key] = color
    return color


def sdr_filter_for(src: Path, *, settings: Settings, cancel: CancelToken | None = None) -> str | None:
    """HDR -> SDR filter chain for ``src`` (None for SDR sources or when FFmpeg cannot convert)."""
    tags = video_color(src, settings=settings, cancel=cancel)
    if not tags.is_hdr:
        return None
    chain = sdr_conversion(settings.ffmpeg)
    return f"{tags.hdr_input_tags()},{chain}" if chain else None
