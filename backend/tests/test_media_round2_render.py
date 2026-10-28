"""Round-2 media review regressions for the renderer and still frames (real FFmpeg on synthetic media).

* finding 0: caption states switch on the first frame at or after their start (not the nearest frame),
  also for times that do not fall on the output frame grid (25 and 29.97 fps);
* finding 2: preview/thumbnail stills ignore a pixel aspect outside 1/4..4 exactly like probe and the render;
* (C): near-standard output frame rates are snapped to the exact standard rational;
* (B): renderer side of per-shot framing (crop plan ``fit_spans`` -> fit picture overlaid inside the spans).
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from conftest import requires_ffmpeg

from clipper.config import Settings
from clipper.framing.analyze import fit_plan
from clipper.media.encoders import cpu_encoder
from clipper.media.frames import extract_frame
from clipper.media.probe import probe
from clipper.models import CaptionStyle, CaptionWord, ClipEdit, Cue, FramingPlan, MediaInfo
from clipper.procs import run
from clipper.render.caption_image import render_caption_track
from clipper.render import renderer
from clipper.render.renderer import build_filter_graph, output_fps, render_clip


def ffmpeg(*args: str) -> None:
    run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", *args], timeout=180)


def media_info(*, fps: float = 25.0, width: int = 1280, height: int = 720) -> MediaInfo:
    return MediaInfo(duration=10.0, size_bytes=1, container="mov,mp4", has_video=True, has_audio=True, width=width,
                     height=height, coded_width=width, coded_height=height, fps=fps, video_codec="h264",
                     audio_codec="aac")


def cue(text: str, a: float, b: float, idx: int = 1) -> Cue:
    wid = f"w{idx:06d}"
    return Cue(id=wid, first_word_id=wid, last_word_id=wid, text=text,
               words=[CaptionWord(word_id=wid, text=text, rel_start=a, rel_end=b)], rel_start=a, rel_end=b,
               highlight=False)


def crop_plan(**kw: Any) -> FramingPlan:
    data: dict[str, Any] = dict(version=3, layout="crop", src_start=2.0, src_end=8.0, src_w=1280, src_h=720,
                                crop_w=404, crop_h=720, crop_y=0, keyframes=[(0.0, 438.0)], reason="t",
                                detector="none")
    data.update(kw)
    return FramingPlan(**data)


@pytest.fixture
def ff(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path / "data", hw_encode="off")


def render(src: Path, media: MediaInfo, edit: ClipEdit, plan: FramingPlan, cues: list[Cue], tmp: Path,
           settings: Settings, name: str) -> tuple[Path, dict[str, Any]]:
    out = tmp / name / "clip.mp4"
    info = render_clip(src=src, media=media, edit=edit, plan=plan, cues=cues, out_mp4=out,
                       work_dir=tmp / f"work_{name}", settings=settings, cancel=None, encoder=cpu_encoder(settings))
    return out, info


def video_stream(path: Path) -> dict[str, Any]:
    res = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-count_frames", "-of", "json",
               str(path)], timeout=60)
    return json.loads(res.stdout.decode())["streams"][0]


# ----------------------------------------------------------------------------- (C) standard frame rates
@pytest.mark.parametrize("fps,want", [
    (19001 / 317, Fraction(60000, 1001)),          # an averaged 59.94 source (was output as 19001/317)
    (59.948, Fraction(60000, 1001)),
    (29.97, Fraction(30000, 1001)),
    (29.978, Fraction(30000, 1001)),
    (23.976, Fraction(24000, 1001)),
    (23.98, Fraction(24000, 1001)),
    (24.006, Fraction(24)),
    (24.993, Fraction(25)),
    (30.008, Fraction(30)),
    (49.994, Fraction(50)),
    (59.999, Fraction(60)),
    (119.88, Fraction(60)),                        # capped, then exactly 60
])
def test_near_standard_rates_snap_to_the_exact_standard(fps: float, want: Fraction) -> None:
    assert output_fps(media_info(fps=fps)) == want


def test_other_rates_are_kept() -> None:
    assert output_fps(media_info(fps=47.5)) == Fraction(95, 2)
    assert output_fps(media_info(fps=26.0)) == Fraction(26)
    assert output_fps(media_info(fps=29.95)) == Fraction(599, 20)          # 0.02 from 29.97: not snapped
    assert output_fps(media_info(fps=12.0)) == Fraction(30)                # low rates: unchanged rule


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_render_of_an_odd_average_rate_is_tagged_with_the_standard_rate(ff: Settings, tmp_path: Path) -> None:
    src = tmp_path / "ntsc60.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=60000/1001:duration=3", "-f", "lavfi", "-i", "sine=d=3",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src))
    media = probe(src, settings=ff).model_copy(update={"fps": 19001 / 317})   # as an averaged probe reports it
    out, info = render(src, media, ClipEdit(src_start=0.5, src_end=2.0, captions_enabled=False),
                       fit_plan(0.5, 2.0, media.width, media.height, reason="t"), [], tmp_path, ff, "ntsc")
    stream = video_stream(out)
    assert stream["r_frame_rate"] == "60000/1001" and stream["avg_frame_rate"] == "60000/1001"
    assert int(stream["nb_read_frames"]) == 90                                # 1.5 s at 59.94
    assert info["fps"] == pytest.approx(59.94, abs=0.001)


# ----------------------------------------------------------------------------- finding 0: caption sampling
@pytest.mark.ffmpeg
@requires_ffmpeg
@pytest.mark.parametrize("rate,fps,expected", [
    ("25", 25.0, [(4, 15), (26, 35), (51, 62)]),
    ("30000/1001", 30000 / 1001, [(5, 18), (31, 42), (61, 74)]),
])
def test_caption_states_appear_on_the_first_frame_at_or_after_their_start(
        ff: Settings, tmp_path: Path, rate: str, fps: float, expected: list[tuple[int, int]]) -> None:
    src = tmp_path / "black.mp4"
    ffmpeg("-f", "lavfi", "-i", f"color=c=black:s=640x360:r={rate}:d=8", "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", str(src))
    media = probe(src, settings=ff)
    edit = ClipEdit(src_start=3.517, src_end=6.517)
    cues = [cue("One", 0.137, 0.611, 1), cue("Two", 1.019, 1.4333, 2), cue("Three", 2.0171, 2.5009, 3)]
    out, _ = render(src, media, edit, fit_plan(3.517, 6.517, media.width, media.height, reason="t"), cues, tmp_path,
                    ff, "cap")
    band = render_caption_track(cues, edit.style, tmp_path / "band", duration=3.0)
    assert band is not None
    raw = run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(out), "-vf",
               f"crop=1080:{band.h}:0:{band.y},scale=270:-2", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
              timeout=60).stdout
    stream = video_stream(out)
    n_frames = int(stream["nb_read_frames"])
    assert n_frames == int(np.ceil(3.0 * fps)) and Fraction(stream["r_frame_rate"]) == Fraction(rate)  # still CFR
    frames = np.frombuffer(raw, np.uint8).reshape(n_frames, -1)
    lit = [i for i, f in enumerate(frames) if f.max() > 128]
    runs: list[tuple[int, int]] = []
    for i in lit:
        if runs and i == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], i)
        else:
            runs.append((i, i))
    # Frame n (at n/fps) shows a cue exactly when start <= n/fps < end.
    want = [(int(np.ceil(c.rel_start * fps - 1e-9)), int(np.ceil(c.rel_end * fps - 1e-9)) - 1) for c in cues]
    assert want == expected
    assert runs == expected


# ----------------------------------------------------------------------------- finding 2: bogus SAR stills
@pytest.mark.ffmpeg
@requires_ffmpeg
@pytest.mark.parametrize("sar", ["5/1", "1/5"])
def test_stills_ignore_an_out_of_range_pixel_aspect_like_probe(ff: Settings, tmp_path: Path, sar: str) -> None:
    src = tmp_path / "sar.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=3", "-vf", f"setsar={sar}",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src))
    stream = video_stream(src)
    assert stream["sample_aspect_ratio"] == sar.replace("/", ":")       # the bogus aspect really is stored
    media = probe(src, settings=ff)
    assert (media.width, media.height) == (640, 360)                     # probe ignores it ...
    jpg = extract_frame(src, 1.0, tmp_path / "f.jpg", max_width=640, settings=ff)
    with Image.open(jpg) as img:
        assert img.size == (640, 360)                                    # ... and so do the stills


# ----------------------------------------------------------------------------- (B) fit spans in the graph
SOURCE_25 = "fps=25/1:start_time=0,tpad=stop_mode=clone:stop=-1,scale=out_range=tv,format=yuv420p"


def test_crop_graph_overlays_the_fit_picture_inside_fit_spans_before_captions(tmp_path: Path) -> None:
    track = render_caption_track([cue("Hi", 0.5, 1.0)], CaptionStyle(), tmp_path, duration=6.0)
    assert track is not None
    plan = crop_plan(fit_spans=[(3.0, 4.0), (1.0, 2.5)])
    graph = build_filter_graph(plan, media_info(), clip_src_start=2.0, track=track, fps=Fraction(25))
    chains = graph.split(";")
    assert chains[0] == f"[0:v]{SOURCE_25},split[cropin][fitin]"
    assert chains[1].startswith("[cropin]crop=w=404:h=720:x='clip(438.0000,0,iw-ow)':y=0,") \
        and chains[1].endswith("[crop]")
    assert chains[2].startswith("[fitin]split[a][b]") and chains[5] == "[bg][fg]overlay=(W-w)/2:(H-h)/2[fit]"
    assert chains[6] == ("[crop][fit]overlay=x=0:y=0:enable='gte(t,1.0000)*lt(t,2.5000)+gte(t,3.0000)*lt(t,4.0000)'"
                         "[base]")
    # Captions are composited on the finished picture, given a fine time base first (finding 0).
    assert chains[7:] == ["[1:v]format=rgba[cap]", "[base]settb=AVTB[main]",
                          f"[main][cap]overlay=x=0:y={track.y}:eof_action=pass:format=auto[vcap]",
                          "[vcap]format=yuv420p[v]"]


def test_fit_spans_follow_the_clip_start_like_keyframes() -> None:
    plan = crop_plan(fit_spans=[(0.0, 1.0), (3.0, 6.0)])
    assert renderer.clip_fit_spans(plan, clip_src_start=2.0) == [(0.0, 1.0), (3.0, 6.0)]
    assert renderer.clip_fit_spans(plan, clip_src_start=2.5) == [(0.0, 0.5), (2.5, 5.5)]
    assert renderer.clip_fit_spans(plan, clip_src_start=3.5) == [(1.5, 4.5)]            # first span is before the clip
    graph = build_filter_graph(plan, media_info(), clip_src_start=2.5, track=None, fps=Fraction(25))
    assert "enable='gte(t,0.0000)*lt(t,0.5000)+gte(t,2.5000)*lt(t,5.5000)'" in graph
    # No spans, or a fit plan: the graphs are unchanged (no split, no timeline overlay).
    plain = build_filter_graph(crop_plan(), media_info(), clip_src_start=2.0, track=None, fps=Fraction(25))
    assert plain.startswith(f"[0:v]{SOURCE_25},crop=") and "enable" not in plain and "cropin" not in plain
    fit = FramingPlan(**{**crop_plan().model_dump(), "layout": "fit", "fit_spans": [(1.0, 2.0)]})
    assert renderer.clip_fit_spans(fit, clip_src_start=2.0) == []
    assert "enable" not in build_filter_graph(fit, media_info(), clip_src_start=2.0, track=None, fps=Fraction(25))
