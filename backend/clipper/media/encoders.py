"""H.264 encoder selection: hardware only after a real capability test, CPU fallback.

``choose_encoder`` encodes one second of a 1080x1920 test pattern with the hardware
encoder (VideoToolbox on macOS), decodes it back, and only then returns it. Any
failure falls back to libx264, which is always used when ``hw_encode == "off"``.
The renderer additionally retries a failed hardware render once with the CPU encoder.
"""

from __future__ import annotations

import platform
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..errors import MediaError
from ..procs import CancelToken, run


@dataclass(frozen=True)
class EncoderChoice:
    name: str                              # "libx264" | "h264_videotoolbox"
    video_args: list[str] = field(default_factory=list)
    hardware: bool = False
    note: str = ""


_lock = threading.Lock()
_cached: dict[tuple[str, str, str, int], EncoderChoice] = {}


def cpu_encoder(settings: Settings) -> EncoderChoice:
    return EncoderChoice(
        name="libx264",
        video_args=[
            "-c:v", "libx264", "-preset", settings.render_preset, "-crf", str(settings.render_crf),
            "-profile:v", "high", "-level:v", "4.2", "-pix_fmt", "yuv420p",
        ],
        hardware=False,
        note="CPU (libx264)",
    )


def _videotoolbox_encoder() -> EncoderChoice:
    return EncoderChoice(
        name="h264_videotoolbox",
        video_args=[
            "-c:v", "h264_videotoolbox", "-b:v", "10M", "-maxrate", "14M", "-bufsize", "20M",
            "-profile:v", "high", "-pix_fmt", "yuv420p", "-allow_sw", "0",
        ],
        hardware=True,
        note="Hardware (VideoToolbox), passed capability test",
    )


def _test_encoder(settings: Settings, choice: EncoderChoice, cancel: CancelToken | None) -> bool:
    tmpdir = Path(tempfile.mkdtemp(prefix="clipper-enc-"))
    out = tmpdir / "probe.mp4"
    try:
        run(
            [settings.ffmpeg, "-hide_banner", "-nostdin", "-y", "-f", "lavfi",
             "-i", "testsrc2=size=1080x1920:rate=30:duration=1", *choice.video_args,
             "-an", "-movflags", "+faststart", str(out)],
            cancel=cancel, timeout=60, error_message="encoder capability test failed",
        )
        run([settings.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-i", str(out), "-f", "null", "-"],
            cancel=cancel, timeout=60, error_message="encoder capability decode failed")
        return out.stat().st_size > 0
    except MediaError:
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def choose_encoder(settings: Settings, *, cancel: CancelToken | None = None) -> EncoderChoice:
    key = (settings.ffmpeg, settings.hw_encode, settings.render_preset, settings.render_crf)
    with _lock:
        if key in _cached:
            return _cached[key]
    choice = cpu_encoder(settings)
    if settings.hw_encode == "auto" and platform.system() == "Darwin":
        hw = _videotoolbox_encoder()
        if _test_encoder(settings, hw, cancel):
            choice = hw
    with _lock:
        _cached[key] = choice
    return choice
