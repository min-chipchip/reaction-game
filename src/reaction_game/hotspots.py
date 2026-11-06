"""Hotspots and the spawner that places them where this player can reach."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .calibration import Calibration

# Angles for hand targets in degrees, measured counter-clockwise from "right",
# with y pointing up. This covers the sides and overhead, plus a little below
# shoulder height.
HAND_ANGLE_RANGE = (-15.0, 195.0)
# Hand targets sit between these fractions of the player's reach: far enough to
# need a real arm extension, but never out of reach.
HAND_DIST_RANGE = (0.5, 0.95)
# Foot targets: sideways offset from the body centre, in shoulder widths.
FOOT_OFFSET_RANGE = (1.5, 2.5)


@dataclass
class Hotspot:
    id: int
    kind: str  # "hand" | "foot"
    x: float
    y: float
    radius: float
    spawn_t: float
    ttl: float

    def expires_at(self) -> float:
        return self.spawn_t + self.ttl

    def remaining(self, t: float) -> float:
        """Fraction of lifetime left, 1 -> 0."""
        return max(0.0, min(1.0, 1.0 - (t - self.spawn_t) / self.ttl))


@dataclass(frozen=True)
class Anchor:
    """Where the player is now, relative to calibration."""

    x: float
    y: float
    scale: float  # current shoulder width / calibrated shoulder width


class Spawner:
    def __init__(self, calibration: Calibration, frame_size: tuple[int, int],
                 rng: random.Random, attempts: int = 40):
        self.cal = calibration
        self.w, self.h = frame_size
        self.rng = rng
        self.attempts = attempts

    def spawn(self, hid: int, t: float, ttl: float, kind: str, anchor: Anchor,
              existing: list[Hotspot], avoid: list[tuple[float, float]] = (),
              clearance: float = 0.0, strict: bool = False) -> Hotspot | None:
        """Place a target of `kind`.

        `avoid` holds the points where the player's hands or feet (of this kind)
        are now, or are about to be (positions extrapolated from their velocity).
        A target placed within `clearance` px of any of them could be touched
        before the player has even seen it, so such candidates are rejected.
        If no candidate clears that distance: with `strict`, return None so the
        caller can try again next frame; otherwise the candidate with the most
        clearance wins.
        """
        r = self.cal.hotspot_radius * anchor.scale
        best: tuple[float, float, float] | None = None  # (clearance, x, y)
        x = y = 0.0
        for _ in range(self.attempts):
            x, y = self._candidate(kind, anchor, r)
            if not (r <= x <= self.w - r and r <= y <= self.h - r):
                continue
            if any(math.hypot(x - o.x, y - o.y) < 1.3 * (r + o.radius) for o in existing):
                continue
            gap = min((math.hypot(x - ax, y - ay) for ax, ay in avoid), default=math.inf)
            if gap >= clearance:
                return Hotspot(hid, kind, x, y, r, t, ttl)
            if best is None or gap > best[0]:
                best = (gap, x, y)
        if strict:
            return None
        if best is not None:
            return Hotspot(hid, kind, best[1], best[2], r, t, ttl)
        # Last resort: clamp the last candidate into the frame (tiny frame or huge player).
        x = min(max(x, r), self.w - r)
        y = min(max(y, r), self.h - r)
        return Hotspot(hid, kind, x, y, r, t, ttl)

    def _candidate(self, kind: str, a: Anchor, r: float) -> tuple[float, float]:
        if kind == "hand":
            theta = math.radians(self.rng.uniform(*HAND_ANGLE_RANGE))
            d = self.rng.uniform(*HAND_DIST_RANGE) * self.cal.reach * a.scale
            return a.x + d * math.cos(theta), a.y - d * math.sin(theta)
        # foot: on the floor line, out to one side
        side = self.rng.choice((-1.0, 1.0))
        dx = side * self.rng.uniform(*FOOT_OFFSET_RANGE) * self.cal.shoulder_width * a.scale
        floor = self.cal.floor_y if self.cal.floor_y is not None else self.h - r
        floor += a.y - self.cal.center[1]  # follow the player if they shift up or down
        return a.x + dx, min(floor - 0.3 * r, self.h - r)
