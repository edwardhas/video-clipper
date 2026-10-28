"""Final clip rendering with FFmpeg: 9:16 framing, burned-in captions, H.264/AAC MP4.

One FFmpeg run per clip:

* fast seek (``-noaccurate_seek -ss`` before ``-i``) to the keyframe before ``edit.src_start``
  and ``-t`` duration. The frames before the cut point are not dropped by the seek: the
  video chain starts with ``fps=<rate>:start_time=0``, which makes the picture constant
  frame rate starting exactly at clip time 0 from the frame *on screen* at that moment
  (a variable-frame-rate source may hold one frame for seconds), and ``atrim=start=0``
  cuts the audio sample-accurately;
* the source picture is then normalised: held past its last frame (``tpad``) so the
  picture always covers the whole clip, HDR (HLG/PQ) tone-mapped to BT.709
  (:mod:`clipper.media.color`), full-range (``yuvj``/pc) converted to limited range,
  8-bit ``yuv420p``, and non-square pixels (anamorphic DVD/HDV) scaled to square;
* video: crop layout (moving 9:16 window from the ``FramingPlan`` keyframes) or fit layout
  (blurred, darkened fill behind the whole frame), scaled to 1080x1920. A crop plan's
  ``fit_spans`` (shots without a single clear face, e.g. slides) are shown in the fit layout:
  both pictures are built from one split of the prepared source and the fit picture is
  overlaid on the crop inside those clip-relative ``[start, end)`` spans;
* captions: Pillow-rendered PNG states fed through the concat demuxer and overlaid on the
  constant-rate picture (FFmpeg builds without libass/freetype are supported — no
  ``subtitles``/``drawtext``). The picture is given a fine time base first, so a caption
  state appears on the first frame at or after its start instead of the nearest frame;
* audio: 48 kHz stereo AAC with 30 ms / 50 ms fades at the cut points (omitted when the
  source has no audio).

Output frame rate: the source rate capped at 60 fps, snapped to the exact standard rate
within 0.01 fps (24000/1001, 24, 25, 30000/1001, 30, 50, 60000/1001, 60); sources below
23 fps (static-image podcasts, sparse variable-frame-rate screen recordings) render at
30 fps so caption and word-highlight switches are not snapped to a coarse frame grid.

The output is written to a temp file next to the destination, validated (streams, size,
duration, faststart layout, full decode) and only then renamed into place. A failed
hardware encode is retried once with libx264.

The crop window is built here from ``FramingPlan`` fields (keyframes in clip-relative
seconds -> piecewise-linear expression of ``t``), so rendering does not depend on the
framing analysis module.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import Settings
from ..errors import DiskSpaceError, MediaError
from ..media.color import sdr_filter_for
from ..media.encoders import EncoderChoice, cpu_encoder
from ..framing.expr import MAX_CROP_SEGMENTS, crop_points, crop_x_expression  # noqa: F401
from ..models import ClipEdit, Cue, FramingPlan, MediaInfo
from ..procs import CancelToken, run
from ..storage import atomic_path, sha256_file, stable_hash
from .caption_image import FONT_PATH, CaptionTrack, render_caption_track

log = logging.getLogger(__name__)

RENDER_VERSION = 3
OUT_W, OUT_H = 1080, 1920
DEFAULT_FPS = Fraction(30)
MAX_FPS = 60
MIN_FPS = 23.0          # lower (or sparse VFR average) rates render at DEFAULT_FPS; 23.976 stays
# Rates within STANDARD_FPS_TOLERANCE of these are output exactly (an averaged 59.94 source may
# report e.g. 19001/317, which would otherwise tag the output with that odd rational).
STANDARD_FPS = tuple(Fraction(n, d) for n, d in ((24000, 1001), (24, 1), (25, 1), (30000, 1001), (30, 1),
                                                 (50, 1), (60000, 1001), (60, 1)))
STANDARD_FPS_TOLERANCE = 0.01
FADE_IN_S = 0.03
FADE_OUT_S = 0.05
AUDIO_RATE = 48000
THUMB_WIDTH = 360


# ----------------------------------------------------------------------------- hashing
@lru_cache(maxsize=4)
def _font_fingerprint(path: Path = FONT_PATH) -> str:
    try:
        return sha256_file(path)[:16]
    except OSError:
        return "missing"


def render_hash(*, source_sha256: str, edit: ClipEdit, plan: FramingPlan, cues: list[Cue], encoder: str) -> str:
    """Content hash of everything that affects the rendered pixels and audio."""
    cue_part = [
        [c.text, c.rel_start, c.rel_end, c.highlight, [[w.text, w.rel_start, w.rel_end] for w in c.words]]
        for c in cues
    ]
    return stable_hash("render", RENDER_VERSION, source_sha256, edit.model_dump(mode="json"),
                       plan.model_dump(mode="json"), cue_part, encoder, _font_fingerprint())


# ----------------------------------------------------------------------------- filter graph
def _num(v: float) -> str:
    """Fixed-point number for FFmpeg expressions (never exponent notation)."""
    out = f"{v:.4f}"
    return "0.0000" if out == "-0.0000" else out


def _even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


def output_fps(media: MediaInfo) -> Fraction:
    """Source frame rate capped at 60 fps as a rational (30000/1001 for 29.97).

    Rates within 0.01 fps of a standard rate (``STANDARD_FPS``) are snapped to it exactly.
    30 fps when unknown or below ``MIN_FPS``: ``media.fps`` is the *average* rate, which for
    variable-frame-rate screen recordings can be far below the nominal rate, and a 1–8 fps
    output would snap every caption/highlight switch to that coarse grid.
    """
    fps = media.fps
    if not fps or not math.isfinite(fps) or fps < MIN_FPS:
        return DEFAULT_FPS
    fps = min(fps, MAX_FPS)
    for std in STANDARD_FPS:
        if abs(fps - float(std)) <= STANDARD_FPS_TOLERANCE:
            return std
    return Fraction(fps).limit_denominator(1001)


def _square_pixels(media: MediaInfo) -> str:
    """``scale`` to the display size for non-square-pixel sources ("" when pixels are square).

    ``MediaInfo.width/height`` are display dimensions (sample aspect ratio applied); crop and
    keyframe coordinates are in display pixels, so the picture is made square first.
    """
    coded = ((media.coded_height, media.coded_width) if media.rotation in (90, 270)
             else (media.coded_width, media.coded_height))
    if media.width > 0 and media.height > 0 and coded[0] > 0 and coded[1] > 0 \
            and (media.width, media.height) != coded:
        return f"scale={media.width}:{media.height},setsar=1"
    return ""


def _source_chain(media: MediaInfo, fps: Fraction, sdr_filter: str | None) -> str:
    """Filters from the decoded source to a clean constant-rate, SDR, limited-range, square-pixel picture."""
    chain = [
        # Constant frame rate from clip time 0, starting with the frame on screen at the cut
        # (needs -noaccurate_seek): VFR held frames are not lost, captions get every frame.
        f"fps={fps.numerator}/{fps.denominator}:start_time=0",
        # Hold the last frame if the source picture ends before the clip does (output is bounded by -t).
        "tpad=stop_mode=clone:stop=-1",
        # HDR -> BT.709 SDR; otherwise just limited range (full-range "yuvj"/pc sources) 8-bit 4:2:0.
        sdr_filter or "scale=out_range=tv,format=yuv420p",
    ]
    square = _square_pixels(media)
    if square:
        chain.append(square)
    return ",".join(chain)


def _crop_chain(plan: FramingPlan, media: MediaInfo, clip_src_start: float) -> str:
    src_w = media.width or plan.src_w
    src_h = media.height or plan.src_h
    crop_h = _even(min(plan.crop_h if plan.crop_h > 0 else src_h, src_h))
    crop_w = _even(min(plan.crop_w if plan.crop_w > 0 else crop_h * 9 / 16, src_w))
    points = crop_points(plan, clip_src_start=clip_src_start)
    x_expr = crop_x_expression(points) if points else _num((src_w - crop_w) / 2)
    crop_y = max(0, min(int(plan.crop_y), src_h - crop_h))
    # Quotes protect the expression's commas from the filtergraph parser.
    return (f"crop=w={crop_w}:h={crop_h}:x='clip({x_expr},0,iw-ow)':y={crop_y},"
            f"scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1")


def _fit_graph(head: str, out: str = "[base]") -> list[str]:
    """Fit layout: the whole frame over a blurred, darkened fill; ``head`` ends in the picture to split."""
    return [
        f"{head}split[a][b]",
        f"[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,gblur=sigma=12,"
        f"scale={OUT_W}:{OUT_H},eq=brightness=-0.06,setsar=1[bg]",
        f"[b]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1[fg]",
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2{out}",
    ]


def clip_fit_spans(plan: FramingPlan, *, clip_src_start: float) -> list[tuple[float, float]]:
    """A crop plan's fit spans as clip-relative ``[start, end)`` output seconds (sorted, non-empty).

    Like keyframes, span times are relative to ``plan.src_start`` and shifted when the clip
    starts elsewhere.
    """
    if plan.layout != "crop":
        return []
    shift = plan.src_start - clip_src_start
    spans = sorted((max(0.0, float(a) + shift), float(b) + shift) for a, b in plan.fit_spans)
    return [(a, b) for a, b in spans if b - a > 1e-4]


def _fit_span_enable(spans: list[tuple[float, float]]) -> str:
    """Timeline expression true inside any ``[start, end)`` span (``t`` = output seconds)."""
    return "+".join(f"gte(t,{_num(a)})*lt(t,{_num(b)})" for a, b in spans)


def build_filter_graph(plan: FramingPlan, media: MediaInfo, *, clip_src_start: float,
                       track: CaptionTrack | None, fps: Fraction, sdr_filter: str | None = None) -> str:
    """``-filter_complex`` graph producing ``[v]`` (input 0 = source, input 1 = caption concat).

    The frame-rate conversion runs on the source, *before* the framing and the caption overlay:
    every later ``t`` is the output time (frame ``n`` at ``n / fps`` from the clip start), and
    the overlay emits one frame per picture frame, so captions can only switch on those frames.
    ``sdr_filter`` is the HDR -> SDR chain for an HDR source (see :mod:`clipper.media.color`).
    """
    source = _source_chain(media, fps, sdr_filter)
    spans = clip_fit_spans(plan, clip_src_start=clip_src_start)
    if plan.layout == "crop" and spans:
        # Per-shot framing: crop and fit pictures from one split of the prepared source; the fit
        # picture replaces the crop inside the spans (before captions are composited on top).
        parts = [f"[0:v]{source},split[cropin][fitin]",
                 f"[cropin]{_crop_chain(plan, media, clip_src_start)}[crop]",
                 *_fit_graph("[fitin]", "[fit]"),
                 f"[crop][fit]overlay=x=0:y=0:enable='{_fit_span_enable(spans)}'[base]"]
    elif plan.layout == "crop":
        parts = [f"[0:v]{source},{_crop_chain(plan, media, clip_src_start)}[base]"]
    else:
        parts = _fit_graph(f"[0:v]{source},")
    tail = "[base]"
    if track is not None:
        parts.append("[1:v]format=rgba[cap]")
        # framesync works in the main input's time base: at 1/fps it would round each caption
        # state's start to the *nearest* frame (up to half a frame early). With a fine time base a
        # state appears on the first frame at or after its start. settb keeps the frame rate (CFR).
        parts.append("[base]settb=AVTB[main]")
        parts.append(f"[main][cap]overlay=x={track.x}:y={track.y}:eof_action=pass:format=auto[vcap]")
        tail = "[vcap]"
    parts.append(f"{tail}format=yuv420p[v]")
    return ";".join(parts)


def _audio_args(duration: float) -> list[str]:
    fade_out_start = max(0.0, duration - FADE_OUT_S)
    # atrim: the fast seek keeps the audio before the cut point; drop it sample-accurately.
    af = (f"atrim=start=0,aresample={AUDIO_RATE},afade=t=in:st=0:d={FADE_IN_S},"
          f"afade=t=out:st={fade_out_start:.6f}:d={FADE_OUT_S}")
    return ["-map", "0:a:0?", "-af", af, "-c:a", "aac", "-b:a", "160k", "-ac", "2", "-ar", str(AUDIO_RATE)]


def build_render_command(*, settings: Settings, src: Path, media: MediaInfo, src_start: float, duration: float,
                         plan: FramingPlan, track: CaptionTrack | None, encoder: EncoderChoice,
                         out_path: Path, sdr_filter: str | None = None) -> list[str]:
    """FFmpeg argument list (never a shell string) for one clip render."""
    args = [settings.ffmpeg, "-hide_banner", "-nostdin", "-y", "-noaccurate_seek", "-ss", f"{src_start:.6f}",
            "-i", str(src)]
    if track is not None:
        args += ["-f", "concat", "-safe", "0", "-i", str(track.concat_path)]
    graph = build_filter_graph(plan, media, clip_src_start=src_start, track=track, fps=output_fps(media),
                               sdr_filter=sdr_filter)
    args += ["-t", f"{duration:.6f}", "-filter_complex", graph, "-map", "[v]"]
    args += _audio_args(duration) if media.has_audio else ["-an"]
    args += list(encoder.video_args)
    args += ["-map_metadata", "-1", "-map_chapters", "-1", "-movflags", "+faststart", "-f", "mp4", str(out_path)]
    return args


# ----------------------------------------------------------------------------- render
def _clip_duration(edit: ClipEdit, media: MediaInfo) -> float:
    src_end = min(edit.src_end, media.duration) if media.duration > 0 else edit.src_end
    duration = src_end - edit.src_start
    if duration <= 0.05:
        raise MediaError("This clip starts at or after the end of the source video. Adjust the clip range.")
    return duration


def _caption_track(edit: ClipEdit, cues: list[Cue], work_dir: Path, duration: float) -> CaptionTrack | None:
    if not edit.captions_enabled or not cues:
        return None
    cap_dir = work_dir / "captions"
    shutil.rmtree(cap_dir, ignore_errors=True)        # drop states from an earlier, different render
    return render_caption_track(cues, edit.style, cap_dir, duration=duration, width=OUT_W, height=OUT_H)


def _is_disk_full(exc: MediaError) -> bool:
    return "no space left on device" in (exc.detail or "").lower()


def render_clip(*, src: Path, media: MediaInfo, edit: ClipEdit, plan: FramingPlan, cues: list[Cue],
                out_mp4: Path, work_dir: Path, settings: Settings, cancel: CancelToken | None,
                encoder: EncoderChoice) -> dict[str, Any]:
    """Render, validate and atomically publish one clip; returns render info for the DB."""
    if not media.has_video:
        raise MediaError("This source has no video track, so a vertical clip cannot be rendered.")
    duration = _clip_duration(edit, media)
    work_dir.mkdir(parents=True, exist_ok=True)
    track = _caption_track(edit, cues, work_dir, duration)
    sdr_filter = sdr_filter_for(src, settings=settings, cancel=cancel)
    attempts = [encoder] + ([cpu_encoder(settings)] if encoder.hardware else [])
    timeout = 300.0 + duration * 30.0
    for i, enc in enumerate(attempts):
        try:
            with atomic_path(out_mp4) as tmp:
                args = build_render_command(settings=settings, src=src, media=media, src_start=edit.src_start,
                                            duration=duration, plan=plan, track=track, encoder=enc, out_path=tmp,
                                            sdr_filter=sdr_filter)
                run(args, cancel=cancel, timeout=timeout, error_message="FFmpeg could not render this clip.")
                info = validate_output(tmp, expected_duration=duration, settings=settings,
                                       expect_audio=media.has_audio, cancel=cancel)
        except MediaError as exc:
            if _is_disk_full(exc):
                raise DiskSpaceError("The disk filled up while rendering this clip. Free some space and retry.",
                                     detail=exc.detail) from exc
            if i + 1 < len(attempts):
                log.warning("render with %s failed (%s); retrying with %s", enc.name, exc.user_message,
                            attempts[i + 1].name)
                continue
            raise
        info.update({
            "encoder": enc.name,
            "render_version": RENDER_VERSION,
            "hdr_to_sdr": sdr_filter is not None,
            "layout": plan.layout,
            "captions": track is not None,
            "caption_images": track.frames if track is not None else 0,
        })
        return info
    raise AssertionError("unreachable")  # pragma: no cover


# ----------------------------------------------------------------------------- validation
def _ffprobe(path: Path, settings: Settings, cancel: CancelToken | None) -> dict[str, Any]:
    res = run([settings.ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
              cancel=cancel, timeout=60, error_message="The rendered clip could not be read back.")
    try:
        data = json.loads(res.stdout.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError("The rendered clip could not be read back.") from exc
    return data if isinstance(data, dict) else {}


def top_level_boxes(path: Path) -> list[tuple[str, int, int]]:
    """(type, offset, size) of the top-level MP4 boxes; MediaError when the file is truncated."""
    total = path.stat().st_size
    boxes: list[tuple[str, int, int]] = []
    off = 0
    with open(path, "rb") as fh:
        while off < total and len(boxes) < 100_000:
            fh.seek(off)
            hdr = fh.read(8)
            if len(hdr) < 8:
                raise MediaError("The rendered clip is truncated.")
            size, kind, header = int.from_bytes(hdr[:4], "big"), hdr[4:8].decode("latin-1"), 8
            if size == 1:
                ext = fh.read(8)
                if len(ext) < 8:
                    raise MediaError("The rendered clip is truncated.")
                size, header = int.from_bytes(ext, "big"), 16
            elif size == 0:                       # box extends to the end of the file
                size = total - off
            if size < header:
                raise MediaError("The rendered clip has an invalid MP4 structure.")
            if off + size > total:
                raise MediaError("The rendered clip is truncated.")
            boxes.append((kind, off, size))
            off += size
    return boxes


def _is_faststart(path: Path) -> bool:
    kinds = [k for k, _, _ in top_level_boxes(path)]
    return "moov" in kinds and "mdat" in kinds and kinds.index("moov") < kinds.index("mdat")


def _rate(value: str | None) -> float:
    try:
        r = Fraction(value or "0")
    except (ValueError, ZeroDivisionError):
        return 0.0
    return float(r)


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _check_streams(data: dict[str, Any], expect_audio: bool | None) -> tuple[list[str], dict[str, Any]]:
    streams = data.get("streams") or []
    videos = [s for s in streams if s.get("codec_type") == "video"]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    problems: list[str] = []
    facts: dict[str, Any] = {"video_codec": None, "audio_codec": None, "width": 0, "height": 0, "fps": 0.0,
                             "v_start": 0.0, "v_dur": None}
    if len(videos) != 1:
        problems.append(f"expected one video stream, found {len(videos)}")
    else:
        v = videos[0]
        facts.update(video_codec=v.get("codec_name"), width=int(v.get("width") or 0),
                     height=int(v.get("height") or 0), fps=_rate(v.get("avg_frame_rate")),
                     v_start=_float(v.get("start_time")) or 0.0, v_dur=_float(v.get("duration")))
        if facts["video_codec"] != "h264":
            problems.append(f"video codec is {facts['video_codec']}, expected h264")
        if (facts["width"], facts["height"]) != (OUT_W, OUT_H):
            problems.append(f"video is {facts['width']}x{facts['height']}, expected {OUT_W}x{OUT_H}")
        if v.get("pix_fmt") != "yuv420p":
            problems.append(f"pixel format is {v.get('pix_fmt')}, expected yuv420p")
    if audios:
        facts["audio_codec"] = audios[0].get("codec_name")
        if len(audios) > 1:
            problems.append(f"expected at most one audio stream, found {len(audios)}")
        if facts["audio_codec"] != "aac":
            problems.append(f"audio codec is {facts['audio_codec']}, expected aac")
        if expect_audio is False:
            problems.append("unexpected audio stream")
    elif expect_audio:
        problems.append("the audio track is missing")
    return problems, facts


def validate_output(path: Path, *, expected_duration: float, settings: Settings, expect_audio: bool | None = None,
                    cancel: CancelToken | None = None) -> dict[str, Any]:
    """Check a rendered MP4; raise MediaError (plain-language, no paths) on any problem.

    ``expect_audio``: True = an AAC track is required, False = none allowed, None = optional.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise MediaError("The rendered clip is missing or empty.")
    size_bytes = path.stat().st_size
    data = _ffprobe(path, settings, cancel)
    problems, facts = _check_streams(data, expect_audio)
    fmt = data.get("format") or {}
    duration = _float(fmt.get("duration"))
    if duration is None:
        durations = [d for d in (_float(s.get("duration")) for s in data.get("streams") or []) if d is not None]
        duration = max(durations) if durations else None
    fps = float(facts["fps"] or 0.0)
    tolerance = max(0.15, 2.0 / fps) if fps > 0 else 0.15
    if duration is None:
        problems.append("its duration is unknown")
    elif abs(duration - expected_duration) > tolerance:
        problems.append(f"it is {duration:.2f} s long, expected {expected_duration:.2f} s")
    # The container duration is the longest stream (usually audio): the picture must cover the clip too.
    v_start, v_dur = float(facts.get("v_start") or 0.0), facts.get("v_dur")
    if v_start > tolerance:
        problems.append(f"the picture starts {v_start:.2f} s late")
    if v_dur is not None and v_start + v_dur < expected_duration - tolerance:
        problems.append(f"the picture covers only {v_start + v_dur:.2f} s of {expected_duration:.2f} s")
    if not _is_faststart(path):
        problems.append("it is not optimised for streaming (index after media data)")
    if problems:
        raise MediaError("The rendered clip failed its quality check: " + "; ".join(problems) + ".")
    run([settings.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
         "-map", "0", "-f", "null", "-"],
        cancel=cancel, timeout=120.0 + expected_duration * 10.0,
        error_message="The rendered clip could not be decoded completely.")
    return {
        "width": facts["width"],
        "height": facts["height"],
        "duration": round(float(duration or 0.0), 3),
        "size_bytes": size_bytes,
        "video_codec": facts["video_codec"],
        "audio_codec": facts["audio_codec"],
        "fps": round(fps, 3),
    }


# ----------------------------------------------------------------------------- thumbnail
def make_thumbnail(mp4: Path, out_jpg: Path, *, t: float, settings: Settings,
                   cancel: CancelToken | None = None) -> None:
    """JPEG still (360 px wide) of the rendered clip at clip time ``t``; written atomically."""
    with atomic_path(out_jpg) as tmp:
        run([settings.ffmpeg, "-hide_banner", "-nostdin", "-y", "-v", "error", "-ss", f"{max(0.0, t):.3f}",
             "-i", str(mp4), "-frames:v", "1", "-vf", f"scale={THUMB_WIDTH}:-2:flags=lanczos", "-q:v", "3",
             "-update", "1", "-f", "image2", str(tmp)],
            cancel=cancel, timeout=60, error_message="Could not create a thumbnail for this clip.")
        if tmp.stat().st_size == 0:
            raise MediaError("Could not create a thumbnail for this clip.")
