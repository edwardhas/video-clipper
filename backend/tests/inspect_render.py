"""Inspect a rendered clip: streams, decode, audio edges, caption sync, face framing, contact sheet.

    cd backend && .venv/bin/python tests/inspect_render.py CLIP.mp4 [--srt CLIP.srt] [--sheet out.png] [--json]

* Streams/decode: ffprobe summary + full decode (``-xerror``).
* Audio: first/last audible moment (silencedetect) and peak level at the start/end (fade check).
* Caption sync (needs --srt): for every SRT cue start, the caption band must change between
  ``start - 0.12 s`` and ``start + 0.12 s`` (a new cue appears) and stay stable mid-cue.
  For speech-like audio, cue starts are compared with the nearest audio onset.
* Framing: YuNet face detections on sampled frames → horizontal centre spread in the 9:16 frame.
* Contact sheet: frames at 10/30/50/70/90 % of the clip with timestamps.
This is a validation tool, not part of the application.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clipper.framing.faces import FaceDetector, detector_available  # noqa: E402

BAND = (0.55, 0.85)   # vertical band (fraction of height) where bottom/middle captions live


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def ffprobe(path: Path) -> dict:
    out = run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    return json.loads(out.stdout)


def frame_at(path: Path, t: float) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(1920, 1080, 3)


def band(img: np.ndarray, region: tuple[float, float]) -> np.ndarray:
    h = img.shape[0]
    return img[int(h * region[0]):int(h * region[1])].astype(np.int16)


def srt_starts(text: str) -> list[tuple[float, float, str]]:
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        m = re.match(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", lines[1])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        cues.append((start, end, " ".join(lines[2:])))
    return cues


def audio_edges(path: Path, duration: float) -> dict:
    out = run(["ffmpeg", "-v", "info", "-i", str(path), "-af", "silencedetect=noise=-40dB:d=0.1", "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", out.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", out.stderr)]
    onsets = sorted(ends + ([0.0] if not starts or starts[0] > 0.05 else []))

    def peak(ss: float, dur: float) -> float | None:
        o = run(["ffmpeg", "-v", "info", "-ss", f"{max(0, ss):.3f}", "-i", str(path), "-t", f"{dur:.3f}", "-vn",
                 "-af", "volumedetect", "-f", "null", "-"])
        m = re.search(r"max_volume: (-?[\d.]+) dB", o.stderr)
        return float(m.group(1)) if m else None

    return {"onsets": [round(x, 3) for x in onsets][:40], "peak_first_50ms_db": peak(0, 0.05),
            "peak_last_50ms_db": peak(duration - 0.05, 0.05)}


def caption_sync(path: Path, cues: list[tuple[float, float, str]], duration: float, region: tuple[float, float]) -> dict:
    changed, stable, checked = 0, 0, 0
    details = []
    for start, end, text in cues:
        if start < 0.15 or start > duration - 0.2:
            continue
        before = band(frame_at(path, start - 0.12), region)
        after = band(frame_at(path, start + 0.12), region)
        diff = float(np.abs(after - before).mean())
        mid_a = band(frame_at(path, start + 0.20), region)
        mid_b = band(frame_at(path, start + 0.20 + 0.04), region)
        checked += 1
        changed += diff > 1.0
        details.append({"cue_start": start, "band_change": round(diff, 2), "text": text[:40]})
        stable += float(np.abs(mid_a - mid_b).mean()) < 6.0
    return {"cues_checked": checked, "changed_at_cue_start": changed, "details": details[:12]}


def highlight_x(img: np.ndarray, region: tuple[float, float]) -> float | None:
    """Centroid x of the amber (#F2A33A) highlight in the caption band, or None if absent."""
    b = band(img, region)
    r, g, bl = b[..., 0], b[..., 1], b[..., 2]
    mask = (r > 200) & (g > 130) & (g < 195) & (bl < 110)
    if mask.sum() < 50:
        return None
    return float(np.nonzero(mask)[1].mean())


def word_highlight(path: Path, onsets: list[float], cues: list[tuple[float, float, str]], duration: float,
                   region: tuple[float, float]) -> dict:
    """At each audio onset inside a cue (not its first word) the amber highlight must jump to the next
    word (centroid moves > 15 px across onset ±60 ms) and stay put inside the word (+80 → +160 ms).
    Only meaningful for the white/boxed styles (amber highlight)."""
    moved, steady, checked, lags = 0, 0, 0, []
    for o in onsets:
        inside = any(c[0] + 0.1 < o < c[1] - 0.1 for c in cues)
        if not inside or o > duration - 0.2:
            continue
        xa, xb = highlight_x(frame_at(path, o - 0.06), region), highlight_x(frame_at(path, o + 0.06), region)
        if xa is None or xb is None:
            continue
        checked += 1
        moved += abs(xb - xa) > 15
        c1, c2 = highlight_x(frame_at(path, o + 0.08), region), highlight_x(frame_at(path, o + 0.16), region)
        steady += c1 is not None and c2 is not None and abs(c2 - c1) < 3
        # first frame (40 ms steps) at which the highlight has moved, relative to the audio onset
        for step in range(-3, 4):
            x = highlight_x(frame_at(path, o + step * 0.04), region)
            if x is not None and abs(x - xa) > 15:
                lags.append(step * 40)
                break
    return {"onsets_checked": checked, "highlight_moved_at_onset": moved, "steady_within_word": steady,
            "highlight_change_vs_onset_ms": lags}


def face_stats(path: Path, duration: float) -> dict | None:
    if not detector_available():
        return None
    det = FaceDetector()
    centres = []
    for frac in np.linspace(0.05, 0.95, 12):
        img = frame_at(path, duration * float(frac))[:, :, ::-1].copy()   # RGB → BGR
        faces = det.detect(img)
        if faces:
            centres.append(faces[0].cx / img.shape[1])
    if not centres:
        return {"frames_with_face": 0}
    return {"frames_with_face": len(centres), "centre_x_mean": round(float(np.mean(centres)), 3),
            "centre_x_min": round(float(np.min(centres)), 3), "centre_x_max": round(float(np.max(centres)), 3)}


def contact_sheet(path: Path, duration: float, out: Path) -> None:
    fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
    thumbs = []
    for f in fracs:
        img = Image.fromarray(frame_at(path, duration * f)).resize((270, 480))
        ImageDraw.Draw(img).text((8, 8), f"t={duration * f:.2f}s", fill=(255, 255, 0))
        thumbs.append(img)
    sheet = Image.new("RGB", (270 * len(thumbs) + 10 * (len(thumbs) - 1), 480), (40, 20, 35))
    for i, t in enumerate(thumbs):
        sheet.paste(t, (i * 280, 0))
    sheet.save(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4", type=Path)
    ap.add_argument("--srt", type=Path)
    ap.add_argument("--sheet", type=Path)
    ap.add_argument("--band", default="bottom", choices=["bottom", "middle", "top"])
    args = ap.parse_args()
    info = ffprobe(args.mp4)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    duration = float(info["format"]["duration"])
    decode = run(["ffmpeg", "-v", "error", "-xerror", "-i", str(args.mp4), "-f", "null", "-"])
    report: dict = {
        "file": args.mp4.name,
        "video": {k: v.get(k) for k in ("codec_name", "width", "height", "pix_fmt", "avg_frame_rate")},
        "audio": {k: a.get(k) for k in ("codec_name", "sample_rate", "channels")} if a else None,
        "duration": round(duration, 3),
        "decode_ok": decode.returncode == 0 and not decode.stderr.strip(),
        "audio_edges": audio_edges(args.mp4, duration) if a else None,
        "faces": face_stats(args.mp4, duration),
    }
    region = {"bottom": BAND, "middle": (0.38, 0.62), "top": (0.12, 0.35)}[args.band]
    if args.srt:
        cues = srt_starts(args.srt.read_text("utf-8"))
        report["captions"] = caption_sync(args.mp4, cues, duration, region)
        onsets = (report["audio_edges"] or {}).get("onsets", [])
        if onsets and cues:
            report["captions"]["cue_vs_audio_onset_ms"] = [
                round(1000 * (c[0] - min(onsets, key=lambda o: abs(o - c[0]))), 1) for c in cues[:12]]
            report["captions"]["word_highlight"] = word_highlight(args.mp4, onsets, cues, duration, region)
    if args.sheet:
        contact_sheet(args.mp4, duration, args.sheet)
        report["contact_sheet"] = str(args.sheet)
    print(json.dumps(report, indent=2))
    return 0 if report["decode_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
