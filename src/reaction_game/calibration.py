"""T-pose calibration: measure the player so hotspots are placed relative to their body.

A child 3 m away and an adult 1.5 m away cover very different numbers of
pixels. Measuring shoulder width and arm reach once lets the spawner place
targets that are always reachable, and always a stretch, for this player.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .pose import L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST, Body

# Typical adult proportions: from the shoulder centre, the palm is about 2.0
# shoulder widths away when the arm is straight. Used only by quick calibration.
QUICK_REACH_RATIO = 2.0


@dataclass(frozen=True)
class Calibration:
    center: tuple[float, float]  # shoulder midpoint (px) at calibration time
    shoulder_width: float        # px
    reach: float                 # px, shoulder centre -> palm with the arm extended
    floor_y: float | None        # px, foot level; None if feet were not visible

    @property
    def hotspot_radius(self) -> float:
        return max(28.0, 0.4 * self.shoulder_width)

    @property
    def feet_visible(self) -> bool:
        return self.floor_y is not None


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def is_t_pose(body: Body, min_vis: float = 0.5) -> bool:
    """Both arms out to the sides at shoulder height.

    Left and right are ignored on purpose, so the check works for a mirrored
    image and for a model that swaps sides.
    """
    idx = (L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST)
    if any(body.vis(i) < min_vis for i in idx):
        return False
    sw = body.shoulder_width()
    if sw < 20:
        return False
    cy = body.shoulder_center()[1]
    sx = sorted((body.xy(L_SHOULDER)[0], body.xy(R_SHOULDER)[0]))
    wrists = [body.xy(L_WRIST), body.xy(R_WRIST)]
    if any(abs(wy - cy) > 0.6 * sw for _, wy in wrists):
        return False
    wx = sorted(x for x, _ in wrists)
    return wx[0] < sx[0] - 0.8 * sw and wx[1] > sx[1] + 0.8 * sw


def _floor_y(body: Body, min_vis: float) -> float | None:
    feet = [e for n, e in body.effectors().items() if n.startswith("foot") and e.visibility >= min_vis]
    return max(e.y for e in feet) if len(feet) == 2 else None


class Calibrator:
    def __init__(self, hold_seconds: float = 1.5, min_vis: float = 0.5):
        self.hold_seconds = hold_seconds
        self.min_vis = min_vis
        self._hold_start: float | None = None
        self._samples: list[tuple[tuple[float, float], float, float, float | None]] = []

    def progress(self, t: float) -> float:
        if self._hold_start is None:
            return 0.0
        return min(1.0, (t - self._hold_start) / self.hold_seconds)

    def update(self, t: float, body: Body | None) -> Calibration | None:
        if body is None or not is_t_pose(body, self.min_vis):
            self._hold_start = None
            self._samples.clear()
            return None
        if self._hold_start is None:
            self._hold_start = t
        center = body.shoulder_center()
        hands = [e for n, e in body.effectors().items() if n.startswith("hand")]
        reach = statistics.fmean(_dist(center, (h.x, h.y)) for h in hands)
        self._samples.append((center, body.shoulder_width(), reach, _floor_y(body, self.min_vis)))
        if t - self._hold_start >= self.hold_seconds and len(self._samples) >= 5:
            return self._aggregate()
        return None

    def _aggregate(self) -> Calibration:
        # Medians, so a few glitchy frames during the hold don't skew the result.
        cx = statistics.median(s[0][0] for s in self._samples)
        cy = statistics.median(s[0][1] for s in self._samples)
        floors = [s[3] for s in self._samples if s[3] is not None]
        return Calibration(
            center=(cx, cy),
            shoulder_width=statistics.median(s[1] for s in self._samples),
            reach=statistics.median(s[2] for s in self._samples),
            floor_y=statistics.median(floors) if len(floors) * 2 >= len(self._samples) else None,
        )

    @staticmethod
    def quick(body: Body | None, min_vis: float = 0.5) -> Calibration | None:
        """Skip the T-pose: estimate reach from shoulder width (less accurate)."""
        if body is None or not body.shoulders_visible(min_vis):
            return None
        sw = body.shoulder_width()
        if sw < 20:
            return None
        return Calibration(body.shoulder_center(), sw, QUICK_REACH_RATIO * sw, _floor_y(body, min_vis))
