"""One Euro filter (Casiez, Roussel & Vogel, CHI 2012) for landmark smoothing.

Raw pose landmarks jitter by a few pixels even when the player stands still. A
fixed low-pass filter forces a bad choice: heavy smoothing removes jitter but
makes fast punches lag, and light smoothing keeps responsiveness but leaves
jitter. The One Euro filter adapts its cutoff frequency to the signal's speed:

    cutoff = min_cutoff + beta * |speed|

When the hand is slow, the low cutoff removes jitter. When it is fast, the
cutoff rises and lag drops. `min_cutoff` mainly controls jitter at rest, and
`beta` mainly controls lag during fast motion.
"""
from __future__ import annotations

import math

from .pose import Effector


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.01, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None:
            self._x, self._t = x, t
            return x
        dt = t - self._t
        if dt <= 0:
            return self._x  # duplicate or out-of-order timestamp
        dx = (x - self._x) / dt
        self._dx = self._dx + _alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x = self._x + _alpha(cutoff, dt) * (x - self._x)
        self._t = t
        return self._x


class EffectorSmoother:
    """Applies a One Euro filter to the x and y of each effector.

    Low-visibility samples are passed through without updating the filter, so
    occlusion noise does not pull the filtered position around. If an effector
    has been unseen for longer than `reset_after`, its filter restarts. This
    avoids a slow glide from where the hand was lost to where it reappears.
    """

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.01,
                 min_visibility: float = 0.5, reset_after: float = 0.3):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.min_visibility = min_visibility
        self.reset_after = reset_after
        self._filters: dict[str, tuple[OneEuroFilter, OneEuroFilter]] = {}
        self._last_seen: dict[str, float] = {}

    def update(self, t: float, raw: dict[str, Effector]) -> dict[str, Effector]:
        out: dict[str, Effector] = {}
        for name, e in raw.items():
            if e.visibility < self.min_visibility:
                out[name] = e
                continue
            fx, fy = self._filters.setdefault(
                name, (OneEuroFilter(self.min_cutoff, self.beta), OneEuroFilter(self.min_cutoff, self.beta))
            )
            last = self._last_seen.get(name)
            if last is not None and t - last > self.reset_after:
                fx.reset()
                fy.reset()
            self._last_seen[name] = t
            out[name] = Effector(fx(e.x, t), fy(e.y, t), e.visibility)
        return out
