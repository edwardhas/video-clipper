"""Renderer: pure helpers (hash, crop expression, command) and real FFmpeg renders of synthetic media."""

from __future__ import annotations

import math
import re
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
from PIL import Image

from clipper.config import Settings
from clipper.errors import MediaError
from clipper.media.encoders import EncoderChoice, cpu_encoder
from clipper.models import CaptionStyle, ClipEdit, FramingPlan, MediaInfo, Word
from clipper.procs import run
from clipper.render.caption_image import render_caption_track
from clipper.render.captions import build_cues
from clipper.render.renderer import (
    build_filter_graph,
    build_render_command,
    crop_points,
    crop_x_expression,
    make_thumbnail,
    output_fps,
    render_clip,
    render_hash,
    top_level_boxes,
    validate_output,
)
from conftest import requires_ffmpeg

SRC_START, SRC_END = 2.0, 6.0
CLIP_SECONDS = SRC_END - SRC_START


# ----------------------------------------------------------------------------- shared builders
def crop_plan(**kw: Any) -> FramingPlan:
    data: dict[str, Any] = dict(version=1, layout="crop", src_start=SRC_START, src_end=SRC_END, src_w=1280, src_h=720,
                                crop_w=404, crop_h=720, crop_y=0, keyframes=[(0.0, 438.0)],
                                reason="single clear face", detector="none")
    data.update(kw)
    return FramingPlan(**data)


def fit_plan() -> FramingPlan:
    return FramingPlan(version=1, layout="fit", src_start=SRC_START, src_end=SRC_END, src_w=1280, src_h=720,
                       reason="multiple people", detector="none")


def media_info(path: Path | None = None, *, has_audio: bool = True, duration: float = 8.0, fps: float = 25.0,
               width: int = 1280, height: int = 720) -> MediaInfo:
    return MediaInfo(duration=duration, size_bytes=path.stat().st_size if path else 1, container="mov,mp4",
                     has_video=True, has_audio=has_audio, width=width, height=height, coded_width=width,
                     coded_height=height, fps=fps, video_codec="h264", audio_codec="aac" if has_audio else None)


def fake_words() -> list[Word]:
    spec = [("Clear", 2.5, 3.0), ("ideas.", 3.0, 3.5), ("Every", 4.0, 4.5), ("time.", 4.5, 5.0)]
    return [Word(id=f"w{i + 1:06d}", text=t, raw=t.lower().strip("."), start=s, end=e, segment_id="s00001",
                 chunk_index=0) for i, (t, s, e) in enumerate(spec)]


def clip_edit(**kw: Any) -> ClipEdit:
    return ClipEdit(src_start=SRC_START, src_end=SRC_END, **kw)


# ----------------------------------------------------------------------------- pure helpers
def _eval_expr(expr: str, t: float) -> float:
    env = {"clip": lambda v, lo, hi: min(max(v, lo), hi), "gte": lambda a, b: 1.0 if a >= b else 0.0, "t": t}
    return float(eval(expr, {"__builtins__": {}}, env))  # noqa: S307 - expression built by our own code


def test_crop_expression_is_piecewise_linear_and_held_at_ends() -> None:
    expr = crop_x_expression([(0.0, 100.0), (2.0, 300.0), (4.0, 300.0), (5.0, 100.0)])
    for t, want in [(-1.0, 100), (0.0, 100), (1.0, 200), (2.0, 300), (3.0, 300), (4.5, 200), (5.0, 100), (9.0, 100)]:
        assert _eval_expr(expr, t) == pytest.approx(want, abs=1e-3), t
    assert "e" not in expr.replace("gte", "")                       # never exponent notation


def test_crop_expression_handles_instant_cut() -> None:
    expr = crop_x_expression([(0.0, 100.0), (1.0, 100.0), (1.0, 500.0)])
    assert _eval_expr(expr, 0.99) == pytest.approx(100.0)
    assert _eval_expr(expr, 1.0) == pytest.approx(500.0)


def test_crop_points_shift_to_clip_and_are_bounded() -> None:
    plan = crop_plan(src_start=1.0, keyframes=[(1.0, 10.0), (2.0, 20.0)])
    assert crop_points(plan, clip_src_start=2.0) == [(0.0, 10.0), (1.0, 20.0)]
    many = crop_plan(keyframes=[(i * 0.1, float(i)) for i in range(500)])
    pts = crop_points(many, clip_src_start=SRC_START)
    # Long tracks are reduced by shape (RDP), not by index (review finding 35), so a straight
    # 500-point ramp collapses to its two end points.
    assert len(pts) <= 61 and pts[0] == (0.0, 0.0) and pts[-1] == pytest.approx((49.9, 499.0))
    zigzag = crop_plan(keyframes=[(i * 0.1, float(100 * (i % 2))) for i in range(500)])
    assert len(crop_points(zigzag, clip_src_start=SRC_START)) <= 61      # still bounded when nothing is redundant


def test_output_fps_is_rational_and_capped() -> None:
    assert output_fps(media_info(fps=30000 / 1001)) == Fraction(30000, 1001)
    assert output_fps(media_info(fps=24000 / 1001)) == Fraction(24000, 1001)
    assert output_fps(media_info(fps=25.0)) == Fraction(25)
    assert output_fps(media_info(fps=119.88)) == Fraction(60)
    assert output_fps(media_info(fps=0.0)) == Fraction(30)
    assert output_fps(media_info(fps=math.nan)) == Fraction(30)
    # Low or sparse-VFR average rates no longer drive the output (review finding 19).
    assert output_fps(media_info(fps=1.0)) == Fraction(30)
    assert output_fps(media_info(fps=120 / 19)) == Fraction(30)
    assert output_fps(media_info(fps=15.0)) == Fraction(30)


# The source chain every graph starts with (review findings 2, 18, 19, 21): constant frame rate from
# clip time 0 *before* the caption overlay, last frame held to the clip end, limited-range 8-bit 4:2:0.
SOURCE_25 = "fps=25/1:start_time=0,tpad=stop_mode=clone:stop=-1,scale=out_range=tv,format=yuv420p"


def test_filter_graphs_for_both_layouts(tmp_path: Path) -> None:
    crop = build_filter_graph(crop_plan(), media_info(), clip_src_start=SRC_START, track=None, fps=Fraction(25))
    assert crop.startswith(f"[0:v]{SOURCE_25},crop=w=404:h=720:x='clip(438.0000,0,iw-ow)':y=0,")
    assert "scale=1080:1920:flags=lanczos,setsar=1[base]" in crop
    assert crop.endswith("[base]format=yuv420p[v]")
    cues = build_cues(fake_words(), clip_edit())
    track = render_caption_track(cues, CaptionStyle(), tmp_path, duration=CLIP_SECONDS)
    assert track is not None
    fit = build_filter_graph(fit_plan(), media_info(), clip_src_start=SRC_START, track=track,
                             fps=Fraction(30000, 1001))
    assert fit.startswith("[0:v]fps=30000/1001:start_time=0,tpad=stop_mode=clone:stop=-1,scale=out_range=tv,"
                          "format=yuv420p,split[a][b];[a]scale=270:480:force_original_aspect_ratio=increase,"
                          "crop=270:480,gblur=sigma=12,scale=1080:1920,eq=brightness=-0.06")
    assert "[bg][fg]overlay=(W-w)/2:(H-h)/2[base]" in fit
    assert "[1:v]format=rgba[cap]" in fit
    # Round-2 finding 0: the picture gets a fine time base before the caption overlay, so a caption state
    # shows from the first frame at or after its start (at 1/fps it was rounded to the nearest frame).
    assert f"[base]settb=AVTB[main];[main][cap]overlay=x=0:y={track.y}:eof_action=pass:format=auto[vcap]" in fit
    assert fit.endswith("[vcap]format=yuv420p[v]")
    hdr = build_filter_graph(fit_plan(), media_info(), clip_src_start=SRC_START, track=None, fps=Fraction(25),
                             sdr_filter="SDR_CHAIN")
    assert hdr.startswith("[0:v]fps=25/1:start_time=0,tpad=stop_mode=clone:stop=-1,SDR_CHAIN,split[a][b];")


def test_non_square_pixels_are_made_square_before_framing() -> None:
    hdv = MediaInfo(duration=8.0, size_bytes=1, container="mov,mp4", has_video=True, has_audio=True, width=1920,
                    height=1080, coded_width=1440, coded_height=1080, fps=25.0)
    graph = build_filter_graph(crop_plan(src_w=1920, src_h=1080), hdv, clip_src_start=SRC_START, track=None,
                               fps=Fraction(25))
    assert graph.startswith(f"[0:v]{SOURCE_25},scale=1920:1080,setsar=1,crop=")
    rotated = hdv.model_copy(update={"width": 1080, "height": 1920, "rotation": 90})
    assert "scale=1080:1920,setsar=1,split" in build_filter_graph(fit_plan(), rotated, clip_src_start=SRC_START,
                                                                  track=None, fps=Fraction(25))
    assert "setsar=1,crop" not in build_filter_graph(crop_plan(), media_info(), clip_src_start=SRC_START,
                                                     track=None, fps=Fraction(25))


def test_crop_without_keyframes_is_centred_and_clamped_to_source() -> None:
    plan = crop_plan(keyframes=[], crop_w=0, crop_h=0)
    graph = build_filter_graph(plan, media_info(width=1281, height=721), clip_src_start=SRC_START, track=None,
                               fps=Fraction(25))
    # crop height = source height (even), width = 9/16 of it (even), centred
    assert "crop=w=404:h=720:x='clip(438.5000,0,iw-ow)':y=0" in graph


def test_render_command_is_an_argument_list(tmp_path: Path, settings: Settings) -> None:
    out = tmp_path / "o.mp4"
    args = build_render_command(settings=settings, src=Path("/x/src file.mp4"), media=media_info(), src_start=2.0,
                                duration=4.0, plan=crop_plan(), track=None, encoder=cpu_encoder(settings),
                                out_path=out)
    assert all(isinstance(a, str) for a in args)
    assert args[args.index("-ss") + 1] == "2.000000" and args.index("-ss") < args.index("-i")
    # Fast seek keeps the frame on screen at the cut (VFR); fps/atrim in the graphs cut exactly.
    assert args.index("-noaccurate_seek") < args.index("-ss")
    assert args[args.index("-i") + 1] == "/x/src file.mp4"            # one argv element, no shell quoting
    assert args[args.index("-t") + 1] == "4.000000"
    assert args[args.index("-af") + 1] == ("atrim=start=0,aresample=48000,afade=t=in:st=0:d=0.03,"
                                           "afade=t=out:st=3.950000:d=0.05")
    assert ["-map", "0:a:0?"] == args[args.index("0:a:0?") - 1:args.index("0:a:0?") + 1]
    assert "+faststart" in args and args[-1] == str(out) and "libx264" in args
    silent = build_render_command(settings=settings, src=Path("s.mp4"), media=media_info(has_audio=False),
                                  src_start=2.0, duration=4.0, plan=fit_plan(), track=None,
                                  encoder=cpu_encoder(settings), out_path=out)
    assert "-an" in silent and "-af" not in silent and "0:a:0?" not in silent


def test_render_hash_is_stable_and_tracks_inputs() -> None:
    def h(**over: Any) -> str:
        edit = over.pop("edit", clip_edit())
        cues = over.pop("cues", build_cues(fake_words(), edit))
        args: dict[str, Any] = dict(source_sha256="ab" * 32, edit=edit, plan=crop_plan(), cues=cues,
                                    encoder="libx264")
        args.update(over)
        return render_hash(**args)

    base = h()
    assert base == h() and len(base) == 64
    yellow = clip_edit(style=CaptionStyle(color="yellow"))
    assert h(edit=yellow) != base
    assert h(edit=clip_edit(style=CaptionStyle(size="L"))) != base
    assert h(edit=clip_edit(word_overrides={"w000001": "Clean"})) != base
    assert h(edit=ClipEdit(src_start=2.1, src_end=SRC_END)) != base
    assert h(plan=crop_plan(keyframes=[(0.0, 400.0)])) != base
    assert h(encoder="h264_videotoolbox") != base
    assert h(source_sha256="cd" * 32) != base
    assert h(cues=[]) != base


def _box(kind: bytes, payload: bytes = b"", *, size: int | None = None) -> bytes:
    return (size if size is not None else 8 + len(payload)).to_bytes(4, "big") + kind + payload


def test_top_level_boxes_parsing(tmp_path: Path) -> None:
    p = tmp_path / "a.mp4"
    large = (1).to_bytes(4, "big") + b"mdat" + (24).to_bytes(8, "big") + b"\0" * 8
    p.write_bytes(_box(b"ftyp", b"isom0000") + _box(b"moov", b"\0" * 4) + large)
    assert [k for k, _, _ in top_level_boxes(p)] == ["ftyp", "moov", "mdat"]
    p.write_bytes(_box(b"ftyp", b"isom0000") + _box(b"mdat", b"\0" * 4, size=100))
    with pytest.raises(MediaError, match="truncated"):
        top_level_boxes(p)
    p.write_bytes(_box(b"ftyp", b"isom0000") + _box(b"moov", b"", size=4))
    with pytest.raises(MediaError, match="invalid"):
        top_level_boxes(p)


# ----------------------------------------------------------------------------- real FFmpeg
def ffmpeg(*args: str) -> None:
    run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", *args], timeout=120)


def frame_rgb(mp4: Path, t: float) -> np.ndarray:
    res = run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-ss", f"{t:.3f}", "-i", str(mp4),
               "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], timeout=60)
    return np.frombuffer(res.stdout, np.uint8).reshape(1920, 1080, 3).astype(int)


def beep_onset(mp4: Path) -> float:
    res = run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(mp4), "-map", "0:a",
               "-af", "silencedetect=noise=-40dB:d=0.1", "-f", "null", "-"], timeout=60)
    m = re.search(r"silence_end: ([\d.]+)", res.stderr_tail)
    assert m, res.stderr_tail
    return float(m.group(1))


@pytest.fixture(scope="module")
def ff_settings(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Settings]:
    data_dir = tmp_path_factory.mktemp("data")
    yield Settings(_env_file=None, data_dir=data_dir, hw_encode="off")
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """8 s 1280x720 testsrc2 with silent audio and a 1 kHz beep at 3.0–3.2 s (source time)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    path = tmp_path_factory.mktemp("src") / "source.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=25:duration=8",
           "-f", "lavfi", "-i", "aevalsrc=exprs='if(between(t,3.0,3.2),0.5*sin(2*PI*1000*t),0)':s=48000:d=8",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
           "-shortest", str(path))
    yield path
    shutil.rmtree(path.parent, ignore_errors=True)


@pytest.fixture(scope="module")
def renders(source: Path, ff_settings: Settings,
            tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    out_dir = tmp_path_factory.mktemp("renders")
    edit = clip_edit()
    cues = build_cues(fake_words(), edit)
    enc = cpu_encoder(ff_settings)
    media = media_info(source)
    result: dict[str, Any] = {"cues": cues, "edit": edit, "out_dir": out_dir}
    for name, plan, use_cues in [("crop_cap", crop_plan(), cues), ("crop_plain", crop_plan(), []),
                                 ("fit_cap", fit_plan(), cues)]:
        out = out_dir / name / "clip.mp4"
        info = render_clip(src=source, media=media, edit=edit, plan=plan, cues=use_cues, out_mp4=out,
                           work_dir=out_dir / f"work_{name}", settings=ff_settings, cancel=None, encoder=enc)
        result[name] = (out, info)
    track = render_caption_track(cues, edit.style, out_dir / "track", duration=CLIP_SECONDS)
    assert track is not None
    result["band"] = (track.y, track.y + track.h)
    yield result
    shutil.rmtree(out_dir, ignore_errors=True)


@pytest.mark.ffmpeg
@requires_ffmpeg
@pytest.mark.parametrize("name", ["crop_cap", "crop_plain", "fit_cap"])
def test_render_outputs_validate(renders: dict[str, Any], ff_settings: Settings, name: str) -> None:
    out, info = renders[name]
    assert out.is_file()
    assert (info["width"], info["height"]) == (1080, 1920)
    assert info["duration"] == pytest.approx(CLIP_SECONDS, abs=0.1)
    assert info["video_codec"] == "h264" and info["audio_codec"] == "aac"
    assert info["fps"] == pytest.approx(25.0)
    assert info["encoder"] == "libx264" and info["captions"] is (name != "crop_plain")
    again = validate_output(out, expected_duration=CLIP_SECONDS, settings=ff_settings, expect_audio=True)
    assert again["size_bytes"] == out.stat().st_size
    assert [p.name for p in out.parent.iterdir()] == ["clip.mp4"]        # temp file renamed away


@pytest.mark.ffmpeg
@requires_ffmpeg
@pytest.mark.parametrize("name", ["crop_cap", "fit_cap"])
def test_audio_is_cut_sample_accurately(renders: dict[str, Any], name: str) -> None:
    out, _ = renders[name]
    assert beep_onset(out) == pytest.approx(1.0, abs=0.05)     # beep at src 3.0 → rel 1.0


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_captions_visible_only_while_cue_is_on(renders: dict[str, Any]) -> None:
    with_caps, _ = renders["crop_cap"]
    plain, _ = renders["crop_plain"]
    y0, y1 = renders["band"]

    def changed_fraction(t: float, rows: slice) -> float:
        diff = np.abs(frame_rgb(with_caps, t) - frame_rgb(plain, t)).max(axis=2)
        return float((diff[rows] > 60).mean())

    band = slice(y0, y1)
    assert changed_fraction(0.8, band) > 0.03                  # cue 1 is on screen at rel 0.5–1.9
    assert changed_fraction(0.2, band) < 0.002                 # nothing before the first word
    assert changed_fraction(0.8, slice(0, y0 - 10)) < 0.002    # captions stay inside their band


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_fit_layout_has_blurred_fill_and_whole_frame(renders: dict[str, Any]) -> None:
    out, _ = renders["fit_cap"]
    frame = frame_rgb(out, 0.2)
    fg_h = round(1080 * 720 / 1280)                            # 608 rows of sharp foreground
    top = (1920 - fg_h) // 2
    fill = frame[100:top - 20]
    fg = frame[top + 20:top + fg_h - 20]
    # The darkened, blurred fill is smooth; the scaled foreground keeps testsrc2's sharp detail.
    assert np.abs(np.diff(fill, axis=1)).mean() < np.abs(np.diff(fg, axis=1)).mean() / 3


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_moving_crop_follows_keyframes_without_audio(ff_settings: Settings, tmp_path: Path) -> None:
    src = tmp_path / "halves.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=red:s=1280x720:r=25:d=4", "-f", "lavfi",
           "-i", "color=c=blue:s=640x720:r=25:d=4", "-filter_complex", "[0:v][1:v]overlay=x=640",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src))
    plan = FramingPlan(version=1, layout="crop", src_start=0.0, src_end=4.0, src_w=1280, src_h=720, crop_w=404,
                       crop_h=720, keyframes=[(0.5, 0.0), (3.5, 876.0)], reason="single clear face",
                       detector="none")
    out = tmp_path / "out" / "pan.mp4"
    info = render_clip(src=src, media=media_info(src, has_audio=False, duration=4.0), edit=ClipEdit(src_start=0.0,
                       src_end=4.0), plan=plan, cues=[], out_mp4=out, work_dir=tmp_path / "work",
                       settings=ff_settings, cancel=None, encoder=cpu_encoder(ff_settings))
    assert info["audio_codec"] is None and info["duration"] == pytest.approx(4.0, abs=0.1)

    def redness(img: np.ndarray) -> float:
        return float((img[..., 0] - img[..., 2]).mean())

    start, middle, end = frame_rgb(out, 0.2), frame_rgb(out, 2.0), frame_rgb(out, 3.8)
    assert redness(start) > 150 and redness(end) < -150           # all red → all blue
    assert redness(middle[:, :480]) > 150 and redness(middle[:, 600:]) < -150   # x=438: boundary mid-frame
    with pytest.raises(MediaError, match="audio track is missing"):
        validate_output(out, expected_duration=4.0, settings=ff_settings, expect_audio=True)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_failed_hardware_encode_retries_with_cpu(source: Path, ff_settings: Settings, tmp_path: Path) -> None:
    broken_hw = EncoderChoice(name="fake_hw", video_args=["-c:v", "no_such_encoder_xyz"], hardware=True)
    edit = ClipEdit(src_start=2.0, src_end=3.0, captions_enabled=False)
    out = tmp_path / "clips" / "retry.mp4"
    info = render_clip(src=source, media=media_info(source), edit=edit, plan=fit_plan(), cues=[], out_mp4=out,
                       work_dir=tmp_path / "work", settings=ff_settings, cancel=None, encoder=broken_hw)
    assert info["encoder"] == "libx264" and out.is_file()


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_failed_render_leaves_nothing_behind(source: Path, ff_settings: Settings, tmp_path: Path) -> None:
    broken = EncoderChoice(name="broken", video_args=["-c:v", "no_such_encoder_xyz"], hardware=False)
    out_dir = tmp_path / "clips"
    with pytest.raises(MediaError) as exc:
        render_clip(src=source, media=media_info(source), edit=ClipEdit(src_start=2.0, src_end=3.0), plan=fit_plan(),
                    cues=[], out_mp4=out_dir / "x.mp4", work_dir=tmp_path / "work", settings=ff_settings,
                    cancel=None, encoder=broken)
    assert str(tmp_path) not in exc.value.user_message and "/" not in exc.value.user_message
    assert list(out_dir.iterdir()) == []


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_clip_past_source_end_is_rejected(source: Path, ff_settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="end of the source"):
        render_clip(src=source, media=media_info(source), edit=ClipEdit(src_start=8.5, src_end=9.0),
                    plan=fit_plan(), cues=[], out_mp4=tmp_path / "x.mp4", work_dir=tmp_path / "w",
                    settings=ff_settings, cancel=None, encoder=cpu_encoder(ff_settings))


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_validation_rejects_bad_files(renders: dict[str, Any], source: Path, ff_settings: Settings,
                                      tmp_path: Path) -> None:
    good, _ = renders["crop_plain"]
    truncated = tmp_path / "truncated.mp4"
    data = good.read_bytes()
    truncated.write_bytes(data[: int(len(data) * 0.6)])
    with pytest.raises(MediaError) as exc:
        validate_output(truncated, expected_duration=CLIP_SECONDS, settings=ff_settings)
    assert str(tmp_path) not in exc.value.user_message

    moov_last = tmp_path / "moov_last.mp4"
    ffmpeg("-i", str(good), "-map", "0", "-c", "copy", str(moov_last))
    with pytest.raises(MediaError, match="streaming"):
        validate_output(moov_last, expected_duration=CLIP_SECONDS, settings=ff_settings)

    with pytest.raises(MediaError, match="expected 1080x1920"):
        validate_output(source, expected_duration=8.0, settings=ff_settings)
    with pytest.raises(MediaError, match="expected 9.00 s"):
        validate_output(good, expected_duration=9.0, settings=ff_settings)
    with pytest.raises(MediaError, match="missing or empty"):
        validate_output(tmp_path / "nope.mp4", expected_duration=1.0, settings=ff_settings)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_corrupt_media_payload_fails_full_decode(renders: dict[str, Any], ff_settings: Settings,
                                                 tmp_path: Path) -> None:
    good, _ = renders["crop_plain"]
    data = bytearray(good.read_bytes())
    boxes = {k: (off, size) for k, off, size in top_level_boxes(good)}
    off, size = boxes["mdat"]
    rng = np.random.default_rng(7)
    start = off + size // 2                    # keep the first frames intact so probing still succeeds
    data[start: off + size] = rng.integers(0, 256, off + size - start, dtype=np.uint8).tobytes()
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(bytes(data))
    with pytest.raises(MediaError, match="decoded"):
        validate_output(corrupt, expected_duration=CLIP_SECONDS, settings=ff_settings)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_thumbnail(renders: dict[str, Any], ff_settings: Settings, tmp_path: Path) -> None:
    out, _ = renders["crop_cap"]
    thumb = tmp_path / "t" / "thumb.jpg"
    make_thumbnail(out, thumb, t=1.0, settings=ff_settings)
    with Image.open(thumb) as img:
        assert img.format == "JPEG" and img.size == (360, 640)
    assert [p.name for p in thumb.parent.iterdir()] == ["thumb.jpg"]

