"""Per-stage latency and FPS, using rolling windows."""
from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager


def percentile(values, q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class Metrics:
    def __init__(self, window: int = 120):
        self.window = window
        self._stages: dict[str, deque[float]] = {}
        self._frame_times: deque[float] = deque(maxlen=window)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - start) * 1000.0)

    def add(self, name: str, ms: float) -> None:
        self._stages.setdefault(name, deque(maxlen=self.window)).append(ms)

    def tick(self, t: float | None = None) -> None:
        self._frame_times.append(time.monotonic() if t is None else t)

    @property
    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def report(self) -> list[tuple[str, float, float]]:
        """[(stage, p50_ms, p95_ms)] in the order stages were first recorded."""
        return [(n, percentile(v, 0.5), percentile(v, 0.95)) for n, v in self._stages.items()]
