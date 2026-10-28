"""FFmpeg crop-window expressions for :class:`~clipper.models.FramingPlan` keyframes.

Single source of truth used by the renderer (and by ``framing.analyze.crop_filter``).
Keyframe times in a plan are relative to ``plan.src_start``; the renderer seeks with
``-ss`` before ``-i`` so the filter's ``t`` starts at 0 at the clip start.
"""

from __future__ import annotations

from typing import Sequence

from ..models import FramingPlan

MAX_CROP_SEGMENTS = 60


def _num(v: float) -> str:
    """Fixed-point number for FFmpeg expressions (never exponent notation)."""
    out = f"{v:.4f}"
    return "0.0000" if out == "-0.0000" else out


def _vertical_rdp(points: Sequence[tuple[float, float]], eps: float) -> list[int]:
    """Ramer–Douglas–Peucker on a time series using vertical (x) error; returns kept indices."""
    n = len(points)
    if n <= 2:
        return list(range(n))
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        (ta, xa), (tb, xb) = points[a], points[b]
        span = tb - ta
        worst, worst_i = -1.0, -1
        for i in range(a + 1, b):
            t, x = points[i]
            expected = xa if span <= 0 else xa + (xb - xa) * (t - ta) / span
            err = abs(x - expected)
            if err > worst:
                worst, worst_i = err, i
        if worst > eps:
            keep[worst_i] = True
            stack.append((a, worst_i))
            stack.append((worst_i, b))
    return [i for i, k in enumerate(keep) if k]


def simplify_keyframes(points: Sequence[tuple[float, float]], *, max_segments: int = MAX_CROP_SEGMENTS,
                       eps: float = 1.0) -> list[tuple[float, float]]:
    """Drop redundant keyframes (error ≤ ``eps`` px), loosening until ≤ ``max_segments`` remain."""
    pts = list(points)
    if len(pts) <= 2:
        return pts
    tol = max(eps, 1e-6)
    while True:
        kept = [pts[i] for i in _vertical_rdp(pts, tol)]
        if len(kept) - 1 <= max_segments:
            return kept
        tol *= 1.5


def crop_points(plan: FramingPlan, *, clip_src_start: float) -> list[tuple[float, float]]:
    """Plan keyframes as (clip-relative t, x_left) sorted by time, at most MAX_CROP_SEGMENTS segments.

    Keyframe times are relative to ``plan.src_start``; they are shifted when the clip starts
    elsewhere so the window stays attached to the same source moment. Longer tracks are
    reduced by shape (RDP), never by index, so both halves of an instant cut survive.
    """
    shift = plan.src_start - clip_src_start
    pts = sorted((float(t) + shift, float(x)) for t, x in plan.keyframes)
    if len(pts) > MAX_CROP_SEGMENTS + 1:
        pts = simplify_keyframes(pts, max_segments=MAX_CROP_SEGMENTS, eps=1.0)
    return pts


def crop_x_expression(points: list[tuple[float, float]]) -> str:
    """Piecewise-linear x(t) through ``points``, held constant before the first/after the last.

    Written as ``x0 + sum(slope_i * clip(t - t_i, 0, span_i))`` — no nesting, no commas outside
    function calls, valid both as an FFmpeg expression and as Python (for unit tests).
    """
    if not points:
        raise ValueError("no crop keyframes")
    terms = [_num(points[0][1])]
    for (t0, xa), (t1, xb) in zip(points, points[1:]):
        dx = xb - xa
        if abs(dx) < 1e-6:
            continue
        span = t1 - t0
        if span <= 1e-4:                      # instantaneous cut to a new position
            term = f"{_num(abs(dx))}*gte(t,{_num(t0)})"
        else:
            term = f"{_num(abs(dx) / span)}*clip(t-{_num(t0)},0,{_num(span)})"
        terms.append(("-" if dx < 0 else "+") + term)
    return "".join(terms)
