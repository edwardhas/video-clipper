"""Regression tests for the media review findings (real FFmpeg on synthetic media).

Finding numbers refer to the review: 2 full-range sources, 14 MPEG-TS/PS audio timeline,
16 non-square pixels, 18 VFR held frames, 19 low/sparse frame rates, 20 HDR sources,
21 picture shorter than the clip, 38 caption switch quantisation.
"""

from __future__ import annotations

import json
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from conftest import requires_ffmpeg

from clipper.config import Settings
from clipper.errors import MediaError
from clipper.framing.analyze import fit_plan, static_crop_plan
from clipper.media import color
from clipper.media.audio import extract_analysis_audio
from clipper.media.encoders import EncoderChoice, _test_encoder, _videotoolbox_encoder, cpu_encoder
from clipper.media.frames import extract_frame, sample_frames
from clipper.media.probe import probe
from clipper.models import CaptionStyle, ClipEdit, Cue, FramingPlan, MediaInfo, Word
from clipper.procs import run
from clipper.render.caption_image import render_caption_track
from clipper.render.captions import build_cues
from clipper.render.renderer import build_render_command, render_clip, validate_output

pytestmark = [pytest.mark.ffmpeg, requires_ffmpeg]


# ----------------------------------------------------------------------------- helpers
def ffmpeg(*args: str) -> None:
    run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", *args], timeout=180)


def ffprobe_json(path: Path, *extra: str) -> dict[str, Any]:
    res = run(["ffprobe", "-v", "error", *extra, "-of", "json", str(path)], timeout=60)
    return json.loads(res.stdout.decode() or "{}")


def video_stream(path: Path) -> dict[str, Any]:
    data = ffprobe_json(path, "-show_streams", "-count_frames", "-select_streams", "v:0")
    return data["streams"][0]


def frame_rgb(mp4: Path, t: float, size: tuple[int, int] = (1080, 1920)) -> np.ndarray:
    res = run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-ss", f"{t:.3f}", "-i", str(mp4),
               "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], timeout=60)
    w, h = size
    return np.frombuffer(res.stdout, np.uint8).reshape(h, w, 3).astype(int)


def onset(path: Path, *pre: str) -> float:
    res = run(["ffmpeg", "-hide_banner", "-nostdin", *pre, "-i", str(path), "-map", "0:a:0",
               "-af", "silencedetect=noise=-40dB:d=0.1", "-f", "null", "-"], timeout=60)
    m = re.search(r"silence_end: ([\d.]+)", res.stderr_tail)
    assert m, res.stderr_tail
    return float(m.group(1))


def words(spec: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(id=f"w{i + 1:06d}", text=t, raw=t.lower(), start=s, end=e, segment_id="s00001", chunk_index=0)
            for i, (t, s, e) in enumerate(spec)]


def render(src: Path, media: MediaInfo, edit: ClipEdit, plan: FramingPlan, cues: list[Cue], tmp: Path,
           settings: Settings, name: str, encoder: EncoderChoice | None = None) -> tuple[Path, dict[str, Any]]:
    out = tmp / name / "clip.mp4"
    info = render_clip(src=src, media=media, edit=edit, plan=plan, cues=cues, out_mp4=out,
                       work_dir=tmp / f"work_{name}", settings=settings, cancel=None,
                       encoder=encoder or cpu_encoder(settings))
    return out, info


def fit(media: MediaInfo, start: float, end: float) -> FramingPlan:
    return fit_plan(start, end, media.width, media.height, reason="test")


@pytest.fixture
def ff(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path / "data", hw_encode="off")


# ----------------------------------------------------------------------------- 2: full range
def _full_range_sources(tmp: Path) -> list[Path]:
    h264 = tmp / "full.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=4", "-f", "lavfi", "-i", "sine=d=4",
           "-vf", "scale=out_range=pc,format=yuvj420p", "-color_range", "pc", "-c:v", "libx264", "-preset",
           "ultrafast", "-c:a", "aac", str(h264))
    mjpeg = tmp / "full.avi"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=4", "-f", "lavfi", "-i", "sine=d=4",
           "-c:v", "mjpeg", "-q:v", "5", "-pix_fmt", "yuvj420p", "-c:a", "pcm_s16le", str(mjpeg))
    return [h264, mjpeg]


@pytest.mark.parametrize("layout", ["crop", "fit"])
def test_full_range_sources_render_and_validate(ff: Settings, tmp_path: Path, layout: str) -> None:
    for src in _full_range_sources(tmp_path):
        media = probe(src, settings=ff)
        assert media.pix_fmt == "yuvj420p"
        edit = ClipEdit(src_start=1.0, src_end=3.0)
        plan = (static_crop_plan(1.0, 3.0, media.width, media.height, centre_x=None, reason="t")
                if layout == "crop" else fit(media, 1.0, 3.0))
        cues = build_cues(words([("Full", 1.2, 1.6), ("range.", 1.7, 2.2)]), edit)
        out, info = render(src, media, edit, plan, cues, tmp_path, ff, f"{src.suffix[1:]}_{layout}")
        stream = video_stream(out)
        assert stream["pix_fmt"] == "yuv420p" and stream.get("color_range", "tv") in ("tv", "unknown")
        assert info["duration"] == pytest.approx(2.0, abs=0.1)


def test_full_range_levels_are_converted_not_retagged(ff: Settings, tmp_path: Path) -> None:
    src = tmp_path / "gray_full.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=0x282828:s=640x360:r=25:d=3", "-vf", "scale=out_range=pc,format=yuvj420p",
           "-color_range", "pc", "-c:v", "libx264", "-preset", "ultrafast", str(src))
    media = probe(src, settings=ff)
    out, _ = render(src, media, ClipEdit(src_start=0.5, src_end=2.5), fit(media, 0.5, 2.5), [], tmp_path, ff, "g")
    # Levels converted to limited range: full-range Y=40 is stored as Y=16+40*219/255=50, still RGB 40.
    assert video_stream(out).get("color_range") != "pc"
    raw = run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-ss", "1.0", "-i", str(out), "-frames:v", "1",
               "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"], timeout=60).stdout
    assert int(np.frombuffer(raw, np.uint8)[960 * 1080 + 540]) == pytest.approx(50, abs=2)
    assert frame_rgb(out, 1.0)[960, 540].mean() == pytest.approx(40, abs=3)


@pytest.mark.skipif(platform.system() != "Darwin", reason="VideoToolbox is macOS-only")
def test_full_range_source_renders_with_videotoolbox(ff: Settings, tmp_path: Path) -> None:
    hw = _videotoolbox_encoder()
    if not _test_encoder(ff, hw, None):
        pytest.skip("VideoToolbox encoder not usable here")
    src = _full_range_sources(tmp_path)[0]
    media = probe(src, settings=ff)
    out, info = render(src, media, ClipEdit(src_start=1.0, src_end=3.0), fit(media, 1.0, 3.0), [], tmp_path, ff,
                       "vt", encoder=hw)
    assert info["encoder"] == "h264_videotoolbox"                 # no fallback to the CPU retry
    assert video_stream(out)["pix_fmt"] == "yuv420p"


# ----------------------------------------------------------------------------- 14: TS/PS audio timeline
def _late_audio_source(tmp: Path, ext: str) -> Path:
    """testsrc2 + a 1 kHz beep at 5.0 s of an audio stream that starts ~0.5 s after the video."""
    out = tmp / f"late.{ext}"
    vc = ["-c:v", "mpeg2video", "-q:v", "4"] if ext == "mpg" else ["-c:v", "libx264", "-preset", "ultrafast",
                                                                    "-pix_fmt", "yuv420p"]
    ac = ["-c:a", "mp2", "-b:a", "128k"] if ext == "mpg" else ["-c:a", "aac", "-b:a", "96k"]
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=12", "-itsoffset", "0.5", "-f", "lavfi",
           "-i", "aevalsrc=exprs='if(between(t,5.0,5.2),0.5*sin(2*PI*1000*t),0)':s=48000:d=11.5",
           "-map", "0:v", "-map", "1:a", *vc, "-g", "25", *ac, str(out))
    return out


@pytest.mark.parametrize("ext", ["ts", "mpg", "mkv", "mp4"])
def test_analysis_wav_and_render_share_the_source_timeline(ff: Settings, tmp_path: Path, ext: str) -> None:
    src = _late_audio_source(tmp_path, ext)
    media = probe(src, settings=ff)
    truth = onset(src, "-copyts") - media.start_time              # beep time measured from the container start
    assert truth == pytest.approx(5.5, abs=0.03)
    wav = tmp_path / "analysis.wav"
    extract_analysis_audio(src, wav, settings=ff)
    out = tmp_path / "r.mp4"
    plan = fit(media, 4.0, 7.0)
    run(build_render_command(settings=ff, src=src, media=media, src_start=4.0, duration=3.0, plan=plan, track=None,
                             encoder=cpu_encoder(ff), out_path=out), timeout=120)
    assert onset(wav) == pytest.approx(truth, abs=0.02)           # transcript/caption/cut times
    assert onset(out) + 4.0 == pytest.approx(truth, abs=0.02)     # what the clip actually plays


# ----------------------------------------------------------------------------- 16: non-square pixels
@pytest.fixture
def anamorphic(tmp_path: Path) -> Path:
    """1440x1080 at SAR 4:3 (16:9 display) with a centred 400x400 *display-pixel* white square."""
    src = tmp_path / "hdv.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=1440x1080:r=25:d=3", "-f", "lavfi", "-i", "sine=d=3",
           "-vf", "drawbox=x=570:y=340:w=300:h=400:color=white:t=fill,setsar=4/3",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src))
    return src


def _white_box(img: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(img.mean(axis=2) > 128)
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def test_anamorphic_source_renders_with_square_pixels(ff: Settings, tmp_path: Path, anamorphic: Path) -> None:
    media = probe(anamorphic, settings=ff)
    assert (media.width, media.height) == (1920, 1080)
    edit = ClipEdit(src_start=0.5, src_end=2.5, captions_enabled=False)
    crop = static_crop_plan(0.5, 2.5, media.width, media.height, centre_x=None, reason="t")
    out, _ = render(anamorphic, media, edit, crop, [], tmp_path, ff, "crop")
    w, h = _white_box(frame_rgb(out, 1.0))
    assert w == pytest.approx(h, rel=0.03) and h == pytest.approx(400 * 1920 / 1080, rel=0.03)
    out, _ = render(anamorphic, media, edit, fit(media, 0.5, 2.5), [], tmp_path, ff, "fit")
    frame = frame_rgb(out, 1.0)
    w, h = _white_box(frame[(1920 - 608) // 2 + 4:(1920 + 608) // 2 - 4])      # sharp 1080x608 foreground
    assert w == pytest.approx(h, rel=0.04) and w == pytest.approx(400 * 1080 / 1920, rel=0.04)


def test_anamorphic_preview_frames_keep_display_proportions(ff: Settings, tmp_path: Path, anamorphic: Path) -> None:
    jpg = extract_frame(anamorphic, 1.0, tmp_path / "f.jpg", max_width=640, settings=ff)
    with Image.open(jpg) as img:
        assert img.size == (640, 360)
        w, h = _white_box(np.asarray(img.convert("RGB")).astype(int))
    assert w == pytest.approx(h, abs=3)
    frames, _ = sample_frames(anamorphic, 0.0, 1.0, fps=2, width=480, settings=ff, cancel=None)   # ffprobe path
    assert frames[0].shape == (270, 480, 3)


# ----------------------------------------------------------------------------- 18: VFR held frames
@pytest.fixture
def vfr(tmp_path: Path) -> Path:
    """VFR: 30 fps content kept only in [0, 0.5) of every 3 s; red for [0,3), blue [3,6), green [6,9)."""
    src = tmp_path / "vfr.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=red:s=320x240:r=30:d=3",
           "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=30:d=3",
           "-f", "lavfi", "-i", "color=c=green:s=320x240:r=30:d=3", "-f", "lavfi", "-i", "sine=d=9",
           "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,select='lt(mod(t,3),0.5)'[v]",
           "-map", "[v]", "-map", "3:a", "-fps_mode", "vfr", "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", "-c:a", "aac", str(src))
    return src


def _dominant(rgb: np.ndarray) -> str:
    return "rgb"[int(np.argmax(rgb))]


def test_vfr_render_starts_with_the_held_frame(ff: Settings, tmp_path: Path, vfr: Path) -> None:
    media = probe(vfr, settings=ff)
    edit = ClipEdit(src_start=1.0, src_end=4.0)
    cues = build_cues(words([("Held", 1.2, 1.6), ("frame", 1.7, 2.1), ("still", 2.2, 2.6), ("captioned.", 2.7, 3.1)]),
                      edit)
    out, info = render(vfr, media, edit, fit(media, 1.0, 4.0), cues, tmp_path, ff, "vfr")
    stream = video_stream(out)
    assert float(stream["start_time"]) == pytest.approx(0.0, abs=0.01)
    assert int(stream["nb_read_frames"]) == pytest.approx(3.0 * info["fps"], abs=2)
    centre = (960, 540)
    assert _dominant(frame_rgb(out, 0.5)[centre]) == "r"           # src 1.5: red frame from 0.467 still shown
    assert _dominant(frame_rgb(out, 2.2)[centre]) == "b"           # src 3.2: blue


def test_vfr_frame_samples_are_labelled_with_the_frame_on_screen(ff: Settings, vfr: Path) -> None:
    frames, times = sample_frames(vfr, 1.0, 4.0, fps=4, width=320, settings=ff, cancel=None)
    assert times == pytest.approx([1.0 + i * 0.25 for i in range(12)])
    for frame, t in zip(frames, times):
        bgr = frame[120, 160]
        assert _dominant(bgr[::-1]) == ("r" if t < 3.0 else "b"), t


# ----------------------------------------------------------------------------- 19: low frame rates
def test_one_fps_source_renders_at_30fps_with_short_highlights(ff: Settings, tmp_path: Path) -> None:
    src = tmp_path / "slides.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=1:duration=8", "-f", "lavfi", "-i", "sine=d=8",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-r", "1", "-c:a", "aac", str(src))
    media = probe(src, settings=ff)
    assert media.fps == pytest.approx(1.0)
    edit = ClipEdit(src_start=2.0, src_end=5.0)
    cues = build_cues(words([("HELLO", 2.5, 3.0), ("THERE", 3.0, 3.5), ("WORLD", 4.0, 4.9)]), edit)
    out, info = render(src, media, edit, fit(media, 2.0, 5.0), cues, tmp_path, ff, "slides")
    assert info["fps"] == pytest.approx(30.0)
    track = render_caption_track(cues, edit.style, tmp_path / "track", duration=3.0)
    assert track is not None
    band = slice(track.y, track.y + track.h)

    def diff(a: float, b: float) -> float:
        return float((np.abs(frame_rgb(out, a)[band] - frame_rgb(out, b)[band]).max(axis=2) > 60).mean())

    # 0.2 and 0.7 show the same source frame; only the HELLO caption (lit 0.5-1.0) differs.
    assert diff(0.7, 0.2) > 0.01
    # 0.7 (HELLO lit) and 1.2 (THERE lit) are different caption states.
    assert diff(0.7, 1.2) > 0.002


# ----------------------------------------------------------------------------- 20: HDR
@pytest.fixture
def hdr_pair(tmp_path: Path, ff: Settings) -> tuple[Path, Path]:
    if color.sdr_conversion(ff.ffmpeg) is None:
        pytest.skip("this FFmpeg build cannot convert HDR to SDR")
    enc = run(["ffmpeg", "-hide_banner", "-encoders"], timeout=30).stdout.decode()
    if "libx265" not in enc:
        pytest.skip("libx265 not available to build an HDR fixture")
    sdr = tmp_path / "sdr.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=25:duration=3", "-f", "lavfi", "-i", "sine=d=3",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-c:a", "aac", str(sdr))
    hlg = tmp_path / "hlg.mp4"
    if "out_transfer" in color._filter_help(ff.ffmpeg, "scale"):
        vf = "scale=out_primaries=bt2020:out_transfer=arib-std-b67:out_color_matrix=bt2020nc:out_range=tv"
    else:
        vf = "zscale=tin=bt709:pin=bt709:min=bt709:rin=tv:t=arib-std-b67:p=bt2020:m=bt2020nc:r=tv"
    ffmpeg("-i", str(sdr), "-vf", f"{vf},format=yuv420p10le", "-c:v", "libx265", "-x265-params", "log-level=error",
           "-preset", "ultrafast", "-tag:v", "hvc1", "-color_primaries", "bt2020", "-color_trc", "arib-std-b67",
           "-colorspace", "bt2020nc", "-c:a", "copy", str(hlg))
    return sdr, hlg


def test_hdr_source_is_tone_mapped_to_sdr_before_captions(ff: Settings, tmp_path: Path,
                                                          hdr_pair: tuple[Path, Path]) -> None:
    sdr, hlg = hdr_pair
    media = probe(hlg, settings=ff)
    edit = ClipEdit(src_start=0.5, src_end=2.5)
    cues = build_cues(words([("Bright", 0.6, 1.0), ("white", 1.1, 1.5)]), edit)
    out, info = render(hlg, media, edit, fit(media, 0.5, 2.5), cues, tmp_path, ff, "hlg")
    assert info["hdr_to_sdr"] is True
    stream = video_stream(out)
    assert stream.get("color_transfer") not in color.HDR_TRANSFERS
    assert stream.get("color_primaries") in (None, "bt709") and stream["pix_fmt"] == "yuv420p"
    ref, _ = render(sdr, probe(sdr, settings=ff), edit, fit(media, 0.5, 2.5), cues, tmp_path, ff, "sdr")
    # The untreated graph (what was rendered before): HLG/BT.2020 code values shown as BT.709.
    raw = tmp_path / "raw.mp4"
    run(build_render_command(settings=ff, src=hlg, media=media, src_start=0.5, duration=2.0,
                             plan=fit(media, 0.5, 2.5), track=None, encoder=cpu_encoder(ff), out_path=raw), timeout=120)
    fg = slice((1920 - 608) // 2 + 10, (1920 + 608) // 2 - 10)
    want = frame_rgb(ref, 0.2)[fg]
    converted = float(np.abs(frame_rgb(out, 0.2)[fg] - want).mean())
    untreated = float(np.abs(frame_rgb(raw, 0.2)[fg] - want).mean())
    assert converted < untreated * 0.75, (converted, untreated)
    # Captions are composited on the SDR picture: white text stays plain SDR white.
    track = render_caption_track(cues, edit.style, tmp_path / "track", duration=2.0)
    assert track is not None
    cap = frame_rgb(out, 0.8)[track.y:track.y + track.h]
    assert cap.max() >= 235


def test_hdr_input_tags_fill_missing_values() -> None:
    hlg = color.VideoColor(transfer="arib-std-b67", primaries=None, space="unknown-x", range=None)
    assert hlg.is_hdr and not color.VideoColor(transfer="bt709").is_hdr
    assert hlg.hdr_input_tags() == ("setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc:"
                                    "range=tv")
    pq = color.VideoColor(transfer="smpte2084", primaries="smpte432", space="bt2020c", range="pc")
    assert pq.hdr_input_tags() == "setparams=color_primaries=smpte432:color_trc=smpte2084:colorspace=bt2020c:range=pc"


def test_hdr_source_with_incomplete_tags_is_converted_like_a_tagged_one(ff: Settings, tmp_path: Path,
                                                                       hdr_pair: tuple[Path, Path]) -> None:
    # Only the transfer is tagged: zscale cannot convert without primaries/matrix, swscale would guess.
    _, hlg = hdr_pair
    partial = tmp_path / "partial.mp4"
    ffmpeg("-i", str(hlg), "-c", "copy", "-bsf:v", "hevc_metadata=colour_primaries=2:matrix_coefficients=2",
           "-color_primaries", "unknown", "-colorspace", "unknown", str(partial))
    tags = color.video_color(partial, settings=ff)
    assert tags.is_hdr and tags.primaries is None and tags.space is None
    full = np.asarray(Image.open(extract_frame(hlg, 1.0, tmp_path / "a.jpg", settings=ff)).convert("RGB")).astype(int)
    part = np.asarray(Image.open(extract_frame(partial, 1.0, tmp_path / "b.jpg", settings=ff)).convert("RGB"))
    assert float(np.abs(part.astype(int) - full).mean()) < 2.0


def test_hdr_preview_frames_are_tone_mapped(ff: Settings, tmp_path: Path, hdr_pair: tuple[Path, Path]) -> None:
    sdr, hlg = hdr_pair
    want = np.asarray(Image.open(extract_frame(sdr, 1.0, tmp_path / "s.jpg", settings=ff)).convert("RGB")).astype(int)
    got = np.asarray(Image.open(extract_frame(hlg, 1.0, tmp_path / "h.jpg", settings=ff)).convert("RGB")).astype(int)
    raw = tmp_path / "raw.jpg"
    ffmpeg("-ss", "1.0", "-i", str(hlg), "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "3", str(raw))
    untreated = np.asarray(Image.open(raw).convert("RGB")).astype(int)
    assert float(np.abs(got - want).mean()) < float(np.abs(untreated - want).mean()) * 0.75


# ----------------------------------------------------------------------------- 21: picture shorter than the clip
def test_validation_rejects_a_picture_that_does_not_cover_the_clip(ff: Settings, tmp_path: Path) -> None:
    short = tmp_path / "short.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=gray:s=1080x1920:r=25:d=3", "-f", "lavfi", "-i", "sine=d=8",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags",
           "+faststart", str(short))
    with pytest.raises(MediaError, match="picture covers only 3.0"):
        validate_output(short, expected_duration=8.0, settings=ff, expect_audio=True)
    late = tmp_path / "late.mp4"
    ffmpeg("-itsoffset", "2", "-f", "lavfi", "-i", "color=c=gray:s=1080x1920:r=25:d=6", "-f", "lavfi",
           "-i", "sine=d=8", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt",
           "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(late))
    with pytest.raises(MediaError, match="picture starts 2.00 s late"):
        validate_output(late, expected_duration=8.0, settings=ff, expect_audio=True)


def test_source_picture_ending_early_holds_its_last_frame(ff: Settings, tmp_path: Path) -> None:
    src = tmp_path / "short_video.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=3", "-f", "lavfi", "-i", "sine=d=8",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src))
    media = probe(src, settings=ff)
    assert media.duration == pytest.approx(8.0, abs=0.1)
    edit = ClipEdit(src_start=2.0, src_end=6.0)
    cues = build_cues(words([("Every", 4.5, 5.0), ("time.", 5.0, 5.5)]), edit)        # after the picture ends
    out, info = render(src, media, edit, fit(media, 2.0, 6.0), cues, tmp_path, ff, "held")
    stream = video_stream(out)
    assert float(stream["duration"]) == pytest.approx(4.0, abs=0.1)
    track = render_caption_track(cues, edit.style, tmp_path / "track", duration=4.0)
    assert track is not None
    band = slice(track.y, track.y + track.h)
    assert float((np.abs(frame_rgb(out, 3.0)[band] - frame_rgb(out, 1.5)[band]).max(axis=2) > 60).mean()) > 0.01


# ----------------------------------------------------------------------------- 38: caption timing precision
def test_caption_states_switch_on_exact_times(ff: Settings, tmp_path: Path) -> None:
    from clipper.models import CaptionWord

    def cue(text: str, a: float, b: float, idx: int) -> Cue:
        return Cue(id=f"w{idx:06d}", first_word_id=f"w{idx:06d}", last_word_id=f"w{idx:06d}", text=text,
                   words=[CaptionWord(word_id=f"w{idx:06d}", text=text, rel_start=a, rel_end=b)], rel_start=a,
                   rel_end=b, highlight=False)

    track = render_caption_track([cue("One", 0.13, 0.5, 1), cue("Two", 0.5, 1.5, 2)], CaptionStyle(), tmp_path / "t",
                                 duration=3.0)
    assert track is not None
    pts = run(["ffprobe", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(track.concat_path),
               "-show_entries", "packet=pts_time", "-of", "csv=p=0"], timeout=60).stdout.decode().split()
    assert [round(float(p), 3) for p in pts[:4]] == [0.0, 0.13, 0.5, 1.5]

    src = tmp_path / "black.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=1280x720:r=30:d=4", "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", str(src))
    media = probe(src, settings=ff)
    edit = ClipEdit(src_start=0.0, src_end=3.0)
    one = [cue("Hello", 0.5, 1.5, 1)]
    out, _ = render(src, media, edit, fit(media, 0.0, 3.0), one, tmp_path, ff, "exact")
    band = render_caption_track(one, edit.style, tmp_path / "t2", duration=3.0)
    assert band is not None
    raw = run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(out), "-vf",
               f"crop=1080:{band.h}:0:{band.y},scale=270:-2", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
              timeout=60).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, len(raw) // 90)
    lit = [i for i, f in enumerate(frames) if f.max() > 128]
    assert (lit[0], lit[-1]) == (15, 44)                    # exactly frames [0.5 s, 1.5 s) at 30 fps

