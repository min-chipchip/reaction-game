"""Hit testing between effectors and hotspots.

At 30 FPS a fast hand swipe can move 150+ px between two frames, enough to
pass straight through a hotspot without either sample landing inside it. So
the test checks the segment from the previous position to the current one
against the circle (a swept test), not just the current point.
"""
from __future__ import annotations

Point = tuple[float, float]


def point_in_circle(p: Point, c: Point, r: float) -> bool:
    dx, dy = p[0] - c[0], p[1] - c[1]
    return dx * dx + dy * dy <= r * r


def segment_circle_hit(p0: Point | None, p1: Point, c: Point, r: float) -> bool:
    """True if the segment p0 -> p1 comes within `r` of `c`.

    `p0=None` falls back to a point test, for example when there is no
    previous sample.
    """
    if p0 is None:
        return point_in_circle(p1, c, r)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    len2 = dx * dx + dy * dy
    if len2 == 0.0:
        return point_in_circle(p1, c, r)
    u = ((c[0] - p0[0]) * dx + (c[1] - p0[1]) * dy) / len2
    u = max(0.0, min(1.0, u))
    return point_in_circle((p0[0] + u * dx, p0[1] + u * dy), c, r)
