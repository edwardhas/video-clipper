"""Automatic vertical framing: face analysis of a source range -> :class:`FramingPlan`.

Pipeline for :func:`analyze_framing`:

1. sample ~4 fps (≤ 240 frames, longer side 480 px) of ``[src_start, src_end)`` with FFmpeg;
2. detect scene cuts from the HSV-histogram correlation between consecutive samples;
3. detect faces (YuNet) and classify every sample as ``single`` (one dominant face),
   ``multi`` or ``none``; each shot takes the majority class of its samples;
4. runs of shots without a single dominant face (``none``: slides, screen recordings;
   ``multi``: wide shots) lasting ≥ ``FIT_SPAN_MIN_S`` become *fit spans*: inside a crop
   plan they are shown full-frame, so a face crop never cuts a slide's text;
5. if shots with a single dominant face cover ≥ 65 % of the clip, or of the time left
   outside the fit spans -> layout ``crop`` with a smoothed, dead-zoned, speed-limited
   horizontal track that resets at cuts (plus the fit spans); otherwise layout ``fit``
   with an honest reason.

Face detection is **not** active-speaker detection. Reasons talk about "a single clear
face" or "multiple people" and never claim to follow the speaker.

Times: samples carry **source** seconds (``src_*``); plan keyframes and fit spans are
**clip-relative** seconds (``rel = src - src_start``) because the renderer seeks with ``-ss``
before ``-i`` so FFmpeg's ``t`` starts at 0 for the clip. A cut lies halfway between the
samples on either side of it, both for the crop track's jump and for fit-span edges.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Literal, Protocol, Sequence

import numpy as np

from ..config import Settings
from ..errors import MediaError
from ..media.frames import sample_frames
from .expr import crop_x_expression, simplify_keyframes
from ..models import FramingChoice, FramingPlan, MediaInfo
from ..procs import CancelToken
from .faces import DETECTOR_NAME, FaceBox, FaceDetector, detector_available

log = logging.getLogger(__name__)

FRAMING_VERSION = 3

# Sampling
SAMPLE_FPS = 4.0
SAMPLE_MAX_FRAMES = 240
SAMPLE_WIDTH = 480             # longer side of the analysed frames (bounds memory: 240 × 480×270×3 B)
# Scene cuts: a new shot starts where the HSV histogram correlation with the previous
# sample falls below this value.
CUT_CORRELATION = 0.5
_HIST_BINS = (16, 8, 8)
_HIST_RANGES = [0, 180, 0, 256, 0, 256]
_HIST_WIDTH = 160
# Face classification
DOMINANCE_RATIO = 1.8          # largest face area ≥ 1.8 × second largest -> single dominant face
MIN_FACE_REL_HEIGHT = 0.05     # ignore faces shorter than 5 % of the frame height (background noise)
CROP_MIN_SHARE = 0.65          # share of clip duration with a single dominant face needed for "crop"
# Per-shot framing: runs of shots without a single dominant face at least this long are shown
# full-frame inside a crop plan; the layout never switches for a shorter stretch than this.
FIT_SPAN_MIN_S = 1.0
# Tracking
MEDIAN_WINDOW = 5              # samples (centred, within a shot)
DEAD_ZONE_FRAC = 0.07          # ± fraction of crop_w around the current centre
MAX_PAN_FRAC_PER_S = 0.35      # max pan speed as a fraction of crop_w per second
MAX_SEGMENTS = 60              # piecewise-linear segments in the final track
SAME_RANGE_TOLERANCE = 0.05    # seconds
_CUT_EPS = 0.002               # seconds between the two keyframes forming an instant jump
_TARGET_ASPECT = 9.0 / 16.0

REASON_CROP = ("A single clear face is on screen for {pct}% of this clip, so the vertical crop "
               "follows that face.")
REASON_CROP_SPANS = ("A single clear face is on screen for {pct}% of this clip, so the vertical crop "
                     "follows that face; shots without a single clear face (for example slides) keep the "
                     "full frame.")
REASON_MIXED = ("A single clear face is on screen for only {pct}% of this clip, so the full frame "
                "is kept.")
REASON_MULTI = "Multiple people are on screen, so the full frame is kept."
REASON_NONE = "No clear face found (e.g. slides), so the full frame is kept."
REASON_UNAVAILABLE = "Face detector unavailable, so the full frame is kept."
REASON_UNREADABLE = "Frames could not be read for face analysis, so the full frame is kept."
REASON_NO_VIDEO = "This source has no video picture, so the full frame is kept."
REASON_MANUAL_FIT = "Full frame kept, as chosen in the editor."
REASON_MANUAL_CROP = "Vertical crop at the position chosen in the editor."
REASON_MANUAL_CROP_ALL = ("Vertical crop for the whole clip, as chosen in the editor; it follows the single "
                          "clear face where there is one.")
REASON_CENTRE_CROP = "Centre crop, as chosen in the editor."
REASON_CENTRE_CROP_NO_FACE = "Centre crop, as chosen in the editor (no single clear face to follow)."
REASON_PENDING_CROP = "Centre crop for now: framing for the new clip range has not been analysed yet " \
                      "(re-analysis pending)."
REASON_PENDING_FIT = "Full frame kept for now: framing for the new clip range has not been analysed yet " \
                     "(re-analysis pending)."

SampleKind = Literal["single", "multi", "none"]


class _Detector(Protocol):
    name: str

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]: ...


@dataclass(frozen=True)
class SampleObs:
    """One analysed sample: source time, class and dominant face centre (source px)."""

    src_t: float
    kind: SampleKind
    face_cx: float | None = None


@dataclass(frozen=True)
class Shot:
    start: int                 # first sample index (inclusive)
    end: int                   # last sample index (exclusive)
    kind: SampleKind           # majority class of the shot's samples
    duration: float            # seconds of the clip covered by this shot


# ----------------------------------------------------------------------------- geometry
def _even_floor(value: float) -> int:
    return max(2, int(math.floor(value)) // 2 * 2)


def _round_even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def crop_geometry(src_w: int, src_h: int) -> tuple[int, int, int]:
    """``(crop_w, crop_h, crop_y)`` of the 9:16 crop window in display pixels.

    Wider than 9:16: full height, ``crop_w = round_even(src_h * 9/16)``. Narrower than 9:16:
    full width, height cut to 16:9-vertical and centred.
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError("source size must be positive")
    if src_w / src_h >= _TARGET_ASPECT:
        crop_w = min(_even_floor(src_w), _round_even(src_h * _TARGET_ASPECT))
        return crop_w, src_h, 0
    crop_w = _even_floor(src_w)
    crop_h = min(src_h, _round_even(crop_w / _TARGET_ASPECT))
    return crop_w, crop_h, max(0, (src_h - crop_h) // 2)


def sample_width(src_w: int, src_h: int) -> int:
    """Analysis frame width: ``SAMPLE_WIDTH`` on the longer side for any orientation."""
    if src_w >= src_h:
        return SAMPLE_WIDTH
    return max(64, int(round(SAMPLE_WIDTH * src_w / src_h)))


def _max_x(src_w: int, crop_w: int) -> float:
    return float(max(0, src_w - crop_w))


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


# ----------------------------------------------------------------------------- plan builders
def _base(src_start: float, src_end: float, src_w: int, src_h: int) -> dict:
    return {"version": FRAMING_VERSION, "src_start": float(src_start), "src_end": float(src_end),
            "src_w": int(src_w), "src_h": int(src_h)}


def fit_plan(src_start: float, src_end: float, src_w: int, src_h: int, *, reason: str, detector: str = "none",
             stats: dict[str, float] | None = None) -> FramingPlan:
    return FramingPlan(**_base(src_start, src_end, src_w, src_h), layout="fit", reason=reason,
                       detector=detector, stats=dict(stats or {}))


def static_crop_plan(src_start: float, src_end: float, src_w: int, src_h: int, *, centre_x: float | None,
                     reason: str, detector: str = "none", stats: dict[str, float] | None = None) -> FramingPlan:
    """Fixed 9:16 crop centred at ``centre_x`` source px (``None`` = frame centre), clamped."""
    crop_w, crop_h, crop_y = crop_geometry(src_w, src_h)
    cx = src_w / 2.0 if centre_x is None else float(centre_x)
    x = _clamp(cx - crop_w / 2.0, 0.0, _max_x(src_w, crop_w))
    return FramingPlan(**_base(src_start, src_end, src_w, src_h), layout="crop", crop_w=crop_w, crop_h=crop_h,
                       crop_y=crop_y, keyframes=[(0.0, round(x, 1))], reason=reason, detector=detector,
                       stats=dict(stats or {}))


# ----------------------------------------------------------------------------- per-sample analysis
def classify_faces(boxes: Sequence[FaceBox], frame_h: int) -> tuple[SampleKind, FaceBox | None]:
    """``single`` with the dominant face, ``multi`` or ``none`` for one sample's detections."""
    min_h = MIN_FACE_REL_HEIGHT * frame_h
    faces = sorted((b for b in boxes if b.h >= min_h and b.w > 0), key=lambda b: b.area, reverse=True)
    if not faces:
        return "none", None
    if len(faces) == 1 or faces[0].area >= DOMINANCE_RATIO * faces[1].area:
        return "single", faces[0]
    return "multi", None


def frame_histogram(frame_bgr: np.ndarray) -> np.ndarray:
    """Normalised HSV histogram used for scene-cut detection."""
    import cv2

    h, w = frame_bgr.shape[:2]
    small = frame_bgr
    if w > _HIST_WIDTH:
        small = cv2.resize(frame_bgr, (_HIST_WIDTH, max(2, round(h * _HIST_WIDTH / w))),
                           interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(_HIST_BINS), _HIST_RANGES)
    return _smooth_histogram(hist.reshape(_HIST_BINS))


def _smooth_histogram(hist: np.ndarray) -> np.ndarray:
    """Spread each bin over its neighbours so small exposure/colour shifts that cross a bin
    edge do not look like a cut. Hue wraps around; saturation/value do not."""
    out = hist.astype(np.float32)
    for axis in range(out.ndim):
        if axis == 0:
            prev, nxt = np.roll(out, 1, axis=0), np.roll(out, -1, axis=0)
        else:
            padded = np.pad(out, [(1, 1) if a == axis else (0, 0) for a in range(out.ndim)], mode="edge")
            prev = np.take(padded, range(0, out.shape[axis]), axis=axis)
            nxt = np.take(padded, range(2, out.shape[axis] + 2), axis=axis)
        out = 0.25 * prev + 0.5 * out + 0.25 * nxt
    total = float(out.sum())
    return (out / total if total > 0 else out).reshape(-1, 1).astype(np.float32)


def histogram_correlation(a: np.ndarray, b: np.ndarray) -> float:
    import cv2

    value = float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
    return 1.0 if math.isnan(value) else value


def detect_cuts(correlations: Sequence[float], *, threshold: float = CUT_CORRELATION) -> list[int]:
    """Sample indices that start a new shot; ``correlations[i]`` compares sample i with i+1."""
    return [i + 1 for i, c in enumerate(correlations) if c < threshold]


def build_shots(observations: Sequence[SampleObs], cuts: Sequence[int], src_end: float) -> list[Shot]:
    """Split samples at ``cuts`` and give each shot its majority class and covered duration.

    Sample i covers ``[t_i, t_{i+1})``; the last sample covers up to ``src_end``. Ties between
    classes resolve conservatively (multi, then none, then single).
    """
    n = len(observations)
    if n == 0:
        return []
    bounds = [0] + sorted({c for c in cuts if 0 < c < n}) + [n]
    shots: list[Shot] = []
    for a, b in zip(bounds, bounds[1:]):
        t_a = observations[a].src_t
        t_b = observations[b].src_t if b < n else max(float(src_end), observations[-1].src_t)
        counts = {"multi": 0, "none": 0, "single": 0}
        for obs in observations[a:b]:
            counts[obs.kind] += 1
        kind: SampleKind = max(("multi", "none", "single"), key=lambda k: counts[k])  # type: ignore[assignment]
        shots.append(Shot(start=a, end=b, kind=kind, duration=max(0.0, t_b - t_a)))
    return shots


# ----------------------------------------------------------------------------- tracking
def median_filter(values: Sequence[float | None], window: int = MEDIAN_WINDOW) -> list[float | None]:
    """Centred median over the non-missing neighbours; missing entries stay missing.

    The window shrinks symmetrically at the ends so it stays centred (no lag at shot edges).
    """
    half = max(0, window // 2)
    n = len(values)
    out: list[float | None] = []
    for i, v in enumerate(values):
        if v is None:
            out.append(None)
            continue
        h = min(half, i, n - 1 - i)
        neigh = [x for x in values[i - h: i + h + 1] if x is not None]
        out.append(float(median(neigh)))
    return out


def track_shot(times: Sequence[float], targets: Sequence[float | None], *, crop_w: int, src_w: int,
               default_centre: float, start_centre: float | None = None) -> list[float]:
    """Crop-window centre (source px) per sample of one shot.

    The window starts on the first available (median-filtered) face centre, moves only when
    the face leaves the dead zone (± ``DEAD_ZONE_FRAC`` × crop_w), then just far enough to put
    the face back on the dead-zone edge, at most ``MAX_PAN_FRAC_PER_S`` × crop_w per second.
    Samples without a face hold the current position. ``start_centre`` (the previous shot's
    final centre) is kept across a cut when the new face is already inside its dead zone, so
    a cut never causes a needless small jump.
    """
    lo, hi = crop_w / 2.0, max(crop_w / 2.0, src_w - crop_w / 2.0)
    filtered = median_filter(targets)
    first = next((v for v in filtered if v is not None), None)
    dead = DEAD_ZONE_FRAC * crop_w
    cam = _clamp(first if first is not None else default_centre, lo, hi)
    if start_centre is not None and abs(cam - _clamp(start_centre, lo, hi)) <= dead:
        cam = _clamp(start_centre, lo, hi)
    speed = MAX_PAN_FRAC_PER_S * crop_w
    centres: list[float] = []
    for i, target in enumerate(filtered):
        if target is not None and i > 0:
            target = _clamp(target, lo, hi)
            gap = target - cam
            if abs(gap) > dead:
                desired = target - math.copysign(dead, gap)
                max_step = speed * max(0.0, times[i] - times[i - 1])
                cam += _clamp(desired - cam, -max_step, max_step)
        centres.append(cam)
    return centres


def _normalise_keyframes(points: Sequence[tuple[float, float]], max_x: float) -> list[tuple[float, float]]:
    """Sorted, rounded (3/1 decimals), clamped, strictly increasing in time."""
    out: list[tuple[float, float]] = []
    for t, x in sorted(points, key=lambda p: p[0]):
        if not (math.isfinite(t) and math.isfinite(x)):
            continue
        t_r = round(max(0.0, float(t)), 3)
        if out and t_r <= out[-1][0]:
            t_r = round(out[-1][0] + 0.001, 3)
        out.append((t_r, round(_clamp(float(x), 0.0, max_x), 1)))
    return out


def track_keyframes(observations: Sequence[SampleObs], shots: Sequence[Shot], *, src_start: float, src_w: int,
                    crop_w: int) -> list[tuple[float, float]]:
    """Clip-relative ``(rel_t, x_left)`` keyframes: per-shot tracks joined by instant jumps."""
    max_x = _max_x(src_w, crop_w)
    centre = src_w / 2.0
    points: list[tuple[float, float]] = []
    last_centre: float | None = None
    for k, shot in enumerate(shots):
        obs = observations[shot.start:shot.end]
        times = [o.src_t for o in obs]
        if shot.kind == "single":
            targets = [o.face_cx if o.kind == "single" else None for o in obs]
        else:
            targets = [None] * len(obs)
        centres = track_shot(times, targets, crop_w=crop_w, src_w=src_w, default_centre=centre,
                             start_centre=last_centre)
        last_centre = centres[-1] if centres else last_centre
        xs = [_clamp(c - crop_w / 2.0, 0.0, max_x) for c in centres]
        if k > 0 and points:
            # Instant jump halfway between the last sample of the previous shot and this one.
            prev_t = observations[shot.start - 1].src_t - src_start
            this_t = times[0] - src_start
            t_cut = (prev_t + this_t) / 2.0
            eps = min(_CUT_EPS, max(0.0, (this_t - prev_t) / 4.0))
            points.append((t_cut - eps / 2.0, points[-1][1]))
            points.append((t_cut + eps / 2.0, xs[0]))
        points.extend((t - src_start, x) for t, x in zip(times, xs))
    if not points:
        return []
    keyframes = _normalise_keyframes(simplify_keyframes(points, max_segments=MAX_SEGMENTS,
                                                         eps=max(1.0, 0.01 * crop_w)), max_x)
    while len(keyframes) >= 2 and keyframes[-1][1] == keyframes[-2][1]:
        keyframes.pop()          # the last value holds after the final keyframe anyway
    return keyframes


# ----------------------------------------------------------------------------- per-shot fit spans
def shot_bounds(observations: Sequence[SampleObs], shots: Sequence[Shot], *, src_start: float,
                src_end: float) -> list[tuple[float, float]]:
    """Clip-relative ``[start, end)`` of every shot: 0, the cut midpoints, then the clip end.

    A cut lies halfway between the last sample of one shot and the first of the next, the same
    instant :func:`track_keyframes` makes the crop window jump.
    """
    total = max(0.0, float(src_end) - float(src_start))
    edges = [0.0]
    for shot in shots[1:]:
        prev_t = observations[shot.start - 1].src_t - src_start
        this_t = observations[shot.start].src_t - src_start
        edges.append(_clamp((prev_t + this_t) / 2.0, edges[-1], total))
    edges.append(total)
    return list(zip(edges, edges[1:]))


def fit_spans(shots: Sequence[Shot], bounds: Sequence[tuple[float, float]], *,
              min_s: float = FIT_SPAN_MIN_S) -> list[tuple[float, float]]:
    """Clip-relative ``[start, end)`` spans to show full-frame inside a crop plan.

    Runs of consecutive shots without a single dominant face (``none``/``multi``) lasting at
    least ``min_s``. Gaps shorter than ``min_s`` between two spans or at either clip edge are
    filled, so the layout never switches for less than ``min_s`` (the full frame is the safe
    choice: it never cuts anything off).
    """
    if not bounds:
        return []
    total = bounds[-1][1]
    runs: list[tuple[float, float]] = []
    prev_fit = False
    for shot, (a, b) in zip(shots, bounds):
        fit = shot.kind != "single"
        if fit and prev_fit:
            runs[-1] = (runs[-1][0], b)
        elif fit:
            runs.append((a, b))
        prev_fit = fit
    spans: list[tuple[float, float]] = []
    for a, b in (r for r in runs if r[1] - r[0] >= min_s):
        if spans and a - spans[-1][1] < min_s:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    if spans and spans[0][0] < min_s:
        spans[0] = (0.0, spans[0][1])
    if spans and total - spans[-1][1] < min_s:
        spans[-1] = (spans[-1][0], total)
    return [(round(a, 3), round(b, 3)) for a, b in spans]


def _cropped_time(shots: Sequence[Shot], bounds: Sequence[tuple[float, float]],
                  spans: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """(seconds shown cropped, of which seconds of single-face shots) outside ``spans``."""
    cropped = single = 0.0
    for shot, (a, b) in zip(shots, bounds):
        outside = (b - a) - sum(max(0.0, min(b, e) - max(a, s)) for s, e in spans)
        cropped += max(0.0, outside)
        if shot.kind == "single":
            single += max(0.0, outside)
    return cropped, single


# ----------------------------------------------------------------------------- decision
def _shares(shots: Sequence[Shot]) -> dict[str, float]:
    total = sum(s.duration for s in shots)
    if total <= 0:
        return {"single": 0.0, "multi": 0.0, "none": 1.0 if shots else 0.0}
    return {k: sum(s.duration for s in shots if s.kind == k) / total for k in ("single", "multi", "none")}


def plan_from_observations(observations: Sequence[SampleObs], cuts: Sequence[int], *, src_start: float,
                           src_end: float, src_w: int, src_h: int, detector: str = DETECTOR_NAME,
                           sample_fps: float = SAMPLE_FPS) -> FramingPlan:
    """Decide crop vs fit from analysed samples (pure; no media access)."""
    shots = build_shots(observations, cuts, src_end)
    shares = _shares(shots)
    stats = {
        "share_single": round(shares["single"], 3),
        "share_multi": round(shares["multi"], 3),
        "share_none": round(shares["none"], 3),
        "shots": float(len(shots)),
        "samples": float(len(observations)),
        "sample_fps": round(float(sample_fps), 4),
    }
    if not shots:
        return fit_plan(src_start, src_end, src_w, src_h, reason=REASON_UNREADABLE, detector=detector, stats=stats)
    bounds = shot_bounds(observations, shots, src_start=src_start, src_end=src_end)
    spans = fit_spans(shots, bounds)
    if spans:
        # Per-shot framing: the spans are shown full-frame anyway, so judge the crop on the rest.
        cropped_s, cropped_single = _cropped_time(shots, bounds, spans)
        crop = cropped_s >= FIT_SPAN_MIN_S and (shares["single"] >= CROP_MIN_SHARE
                                                or cropped_single >= CROP_MIN_SHARE * cropped_s)
    else:
        crop = shares["single"] >= CROP_MIN_SHARE
    if crop:
        crop_w, crop_h, crop_y = crop_geometry(src_w, src_h)
        keyframes = track_keyframes(observations, shots, src_start=src_start, src_w=src_w, crop_w=crop_w)
        pct = int(round(shares["single"] * 100))
        reason = (REASON_CROP_SPANS if spans else REASON_CROP).format(pct=pct)
        return FramingPlan(**_base(src_start, src_end, src_w, src_h), layout="crop", crop_w=crop_w,
                           crop_h=crop_h, crop_y=crop_y, keyframes=keyframes, fit_spans=spans, reason=reason,
                           detector=detector, stats=stats)
    if shares["single"] > 0 and shares["single"] >= max(shares["multi"], shares["none"]):
        reason = REASON_MIXED.format(pct=int(round(shares["single"] * 100)))
    elif shares["multi"] > 0 and shares["multi"] >= shares["none"]:
        reason = REASON_MULTI
    else:
        reason = REASON_NONE
    return fit_plan(src_start, src_end, src_w, src_h, reason=reason, detector=detector, stats=stats)


def observe_samples(frames: Sequence[np.ndarray], times: Sequence[float], *, src_w: int, detector: _Detector,
                    cancel: CancelToken | None = None) -> tuple[list[SampleObs], list[int]]:
    """Run face detection and cut detection over sampled frames (display orientation)."""
    observations: list[SampleObs] = []
    correlations: list[float] = []
    prev_hist: np.ndarray | None = None
    for i, (frame, src_t) in enumerate(zip(frames, times)):
        if cancel is not None and i % 8 == 0:
            cancel.raise_if_cancelled()
        frame_h, frame_w = frame.shape[:2]
        kind, face = classify_faces(detector.detect(frame), frame_h)
        face_cx = face.cx * (src_w / frame_w) if face is not None else None
        observations.append(SampleObs(src_t=float(src_t), kind=kind, face_cx=face_cx))
        hist = frame_histogram(frame)
        if prev_hist is not None:
            correlations.append(histogram_correlation(prev_hist, hist))
        prev_hist = hist
    return observations, detect_cuts(correlations)


def analyze_framing(src: Path, media: MediaInfo, src_start: float, src_end: float, *, settings: Settings,
                    cancel: CancelToken | None) -> FramingPlan:
    """Analyse faces/shots in ``[src_start, src_end)`` (source seconds) and return a plan."""
    if src_end <= src_start:
        raise ValueError("src_end must be after src_start")
    src_w, src_h = int(media.width), int(media.height)
    if not media.has_video or src_w <= 0 or src_h <= 0:
        return fit_plan(src_start, src_end, max(0, src_w), max(0, src_h), reason=REASON_NO_VIDEO)
    if not detector_available():
        return fit_plan(src_start, src_end, src_w, src_h, reason=REASON_UNAVAILABLE)
    try:
        detector = FaceDetector()
    except Exception as exc:  # model unreadable / OpenCV build problem
        log.warning("face detector could not be created: %s", type(exc).__name__)
        return fit_plan(src_start, src_end, src_w, src_h, reason=REASON_UNAVAILABLE)

    sample_end = min(float(src_end), float(media.duration)) if media.duration > 0 else float(src_end)
    if sample_end <= src_start:
        return fit_plan(src_start, src_end, src_w, src_h, reason=REASON_UNREADABLE, detector=detector.name)
    try:
        frames, times = sample_frames(src, src_start, sample_end, fps=SAMPLE_FPS,
                                      width=sample_width(src_w, src_h),
                                      settings=settings, cancel=cancel, max_frames=SAMPLE_MAX_FRAMES,
                                      display_size_hint=(src_w, src_h))
    except MediaError as exc:
        log.warning("framing: frame sampling failed (%s)", exc.detail or exc.user_message)
        return fit_plan(src_start, src_end, src_w, src_h, reason=REASON_UNREADABLE, detector=detector.name)

    sample_fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] \
        else SAMPLE_FPS
    observations, cuts = observe_samples(frames, times, src_w=src_w, detector=detector, cancel=cancel)
    del frames
    plan = plan_from_observations(observations, cuts, src_start=src_start, src_end=src_end, src_w=src_w,
                                  src_h=src_h, detector=detector.name, sample_fps=sample_fps)
    log.info("framing %.2f-%.2f: layout=%s shots=%d samples=%d single=%.2f fit_spans=%d", src_start, src_end,
             plan.layout, int(plan.stats.get("shots", 0)), int(plan.stats.get("samples", 0)),
             plan.stats.get("share_single", 0.0), len(plan.fit_spans))
    return plan


# ----------------------------------------------------------------------------- editor choices
def _same_range(plan: FramingPlan, media: MediaInfo, src_start: float, src_end: float) -> bool:
    return (abs(plan.src_start - src_start) <= SAME_RANGE_TOLERANCE
            and abs(plan.src_end - src_end) <= SAME_RANGE_TOLERANCE
            and plan.src_w == media.width and plan.src_h == media.height)


def _interp(points: Sequence[tuple[float, float]], t: float) -> float:
    ts = [p[0] for p in points]
    xs = [p[1] for p in points]
    return float(np.interp(t, ts, xs))


def _shift_keyframes(points: Sequence[tuple[float, float]], shift: float) -> list[tuple[float, float]]:
    """Re-base keyframes by ``shift`` seconds (``rel_new = rel_old + shift``), starting at 0."""
    if not points or abs(shift) < 1e-9:
        return list(points)
    moved = [(t + shift, x) for t, x in points]
    if moved[0][0] > 0:
        return [(0.0, moved[0][1])] + moved
    head = (0.0, _interp(moved, 0.0))
    return [head] + [p for p in moved if p[0] > 0]


def _shift_spans(spans: Sequence[tuple[float, float]], shift: float, old_total: float,
                 new_total: float) -> list[tuple[float, float]]:
    """Re-base fit spans like keyframes (``rel_new = rel_old + shift``), clipped to the new clip.

    A span that started at the old clip start or ran to the old clip end keeps doing so, so a
    slightly longer range never shows a crop frame at its edges.
    """
    out: list[tuple[float, float]] = []
    for a, b in spans:
        a2 = 0.0 if a <= 1e-3 else max(0.0, a + shift)
        b2 = new_total if b >= old_total - 1e-3 else min(new_total, b + shift)
        if b2 - a2 > 1e-3:
            out.append((round(a2, 3), round(b2, 3)))
    return out


def _rebase(plan: FramingPlan, src_start: float, src_end: float, *, keep_spans: bool = True) -> FramingPlan:
    shift = plan.src_start - src_start
    max_x = _max_x(plan.src_w, plan.crop_w)
    keyframes = _normalise_keyframes(_shift_keyframes(plan.keyframes, shift), max_x) \
        if plan.layout == "crop" else []
    if len(keyframes) > MAX_SEGMENTS + 1:
        # Re-basing an earlier start prepends a (collinear) head point; simplify by shape so
        # the renderer never has to drop a point (which could split an instant-cut pair).
        keyframes = _normalise_keyframes(simplify_keyframes(keyframes, max_segments=MAX_SEGMENTS, eps=1.0), max_x)
    spans = _shift_spans(plan.fit_spans, shift, plan.src_end - plan.src_start, float(src_end) - float(src_start)) \
        if keep_spans and plan.layout == "crop" else []
    return plan.model_copy(update={"src_start": float(src_start), "src_end": float(src_end),
                                   "keyframes": keyframes, "fit_spans": spans})


def plan_for_edit(auto: FramingPlan | None, choice: FramingChoice, media: MediaInfo, src_start: float,
                  src_end: float) -> FramingPlan:
    """The plan to render for the editor's framing ``choice`` over ``[src_start, src_end)``.

    Automatic keyframes (and fit spans) are only reused when ``auto`` covers the same range
    (± 0.05 s); those analysed for a different range are never applied to this one. Fit spans
    belong to the automatic choice: a manual crop crops the whole clip, a manual fit has none.
    """
    src_w, src_h = int(media.width), int(media.height)
    if src_w <= 0 or src_h <= 0:
        return fit_plan(src_start, src_end, max(0, src_w), max(0, src_h), reason=REASON_NO_VIDEO)
    reusable = auto is not None and _same_range(auto, media, src_start, src_end)
    detector = auto.detector if auto is not None else "none"

    if choice.mode == "fit":
        return fit_plan(src_start, src_end, src_w, src_h, reason=REASON_MANUAL_FIT, detector="manual")
    if choice.mode == "crop":
        if choice.crop_x is not None:
            return static_crop_plan(src_start, src_end, src_w, src_h, centre_x=choice.crop_x * src_w,
                                    reason=REASON_MANUAL_CROP, detector="manual")
        if reusable and auto is not None and auto.layout == "crop" and auto.keyframes:
            tracked = _rebase(auto, src_start, src_end, keep_spans=False)
            return tracked.model_copy(update={"reason": REASON_MANUAL_CROP_ALL}) if auto.fit_spans else tracked
        analysed = reusable and auto is not None and auto.detector not in ("none", "manual")
        reason = REASON_CENTRE_CROP_NO_FACE if analysed else REASON_CENTRE_CROP
        return static_crop_plan(src_start, src_end, src_w, src_h, centre_x=None, reason=reason,
                                detector=detector)
    # mode == "auto"
    if reusable and auto is not None:
        return _rebase(auto, src_start, src_end)
    if auto is not None and auto.layout == "crop":
        return static_crop_plan(src_start, src_end, src_w, src_h, centre_x=None, reason=REASON_PENDING_CROP,
                                detector=detector)
    return fit_plan(src_start, src_end, src_w, src_h, reason=REASON_PENDING_FIT, detector=detector)


# ----------------------------------------------------------------------------- FFmpeg expressions
def _num(value: float, decimals: int = 2) -> str:
    text = f"{value:.{decimals}f}"
    return "0" if text.strip("-0.") == "" else text


def crop_x_expr(plan: FramingPlan) -> str:
    """FFmpeg expression of ``t`` (clip-relative seconds) for the crop window's left edge.

    Piecewise linear through the plan's keyframes (shared builder in :mod:`clipper.framing.expr`,
    the same one the renderer uses), constant before the first and after the last keyframe, and
    clamped with ``clip(…,0,src_w-crop_w)``. The expression contains commas: use
    :func:`crop_filter` (which quotes it) when putting it into a filtergraph.
    """
    if plan.layout != "crop" or plan.crop_w <= 0:
        raise ValueError("crop_x_expr needs a crop plan")
    max_x = _max_x(plan.src_w, plan.crop_w)
    points = _normalise_keyframes(plan.keyframes, max_x)
    if len(points) > MAX_SEGMENTS + 1:
        points = _normalise_keyframes(simplify_keyframes(points, max_segments=MAX_SEGMENTS, eps=1.0), max_x)
    if not points:
        return _num(max_x / 2.0)
    if len(points) == 1:
        return _num(points[0][1])
    return f"clip({crop_x_expression(points)},0,{_num(max_x, 0)})"


def crop_filter(plan: FramingPlan) -> str:
    """Filtergraph-safe ``crop=w=..:h=..:x='<expr>':y=..`` for a crop plan.

    The x expression is single-quoted so its commas do not split the filter chain; it never
    contains quotes or colons, so no further escaping is needed.
    """
    expr = crop_x_expr(plan)
    if "'" in expr or ":" in expr:  # pragma: no cover - guarded by construction
        raise ValueError("unexpected characters in crop expression")
    return f"crop=w={plan.crop_w}:h={plan.crop_h}:x='{expr}':y={plan.crop_y}"
