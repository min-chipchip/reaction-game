"""Game rules and state machine.

The game never reads a clock or a camera. Every `update` receives the frame's
timestamp and the tracked body, so tests and replays can drive it with fake
time and recorded or synthetic input.
"""
from __future__ import annotations

import enum
import math
import random
import statistics
from dataclasses import dataclass, field

from .calibration import Calibration, Calibrator
from .collision import segment_circle_hit
from .hotspots import Anchor, Hotspot, Spawner
from .pose import Body, Effector, effector_kind


class State(enum.Enum):
    CALIBRATING = "calibrating"
    COUNTDOWN = "countdown"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"


@dataclass(frozen=True)
class GameConfig:
    duration: float = 60.0
    countdown: float = 3.0
    mode: str = "hands"                       # "hands" | "hands_feet"
    ttl_start: float = 3.0                    # hotspot lifetime at round start...
    ttl_end: float = 1.5                      # ...shrinking linearly to this by round end
    spawn_gap: tuple[float, float] = (0.25, 0.7)
    max_active: int = 2
    second_hotspot_after: float = 15.0        # a single target at a time until then
    base_points: int = 100
    speed_bonus: int = 100                    # extra points for an instant touch, falling to 0 at expiry
    combo_step: float = 0.1
    combo_cap: int = 10
    min_visibility: float = 0.5
    rearm: float = 0.2                        # an effector that just scored can't score again until then
    pause_after: float = 0.75                 # player missing this long -> pause
    foot_probability: float = 0.3
    max_sweep_gap: float = 0.15               # only sweep between samples this close in time
    max_sweep_reach: float = 1.5              # ...and this close in space (x reach): longer jumps are tracking glitches
    touch_tolerance: float = 0.08             # extra hit radius, x shoulder width (effector isn't a point)
    spawn_clearance: float = 0.6              # min distance from a same-kind effector to a new target, x reach
    spawn_lookahead: float = 0.3              # ...also from where that effector will be within this many seconds
    spawn_max_delay: float = 0.5              # no clear spot: retry each frame this long, then take the best spot


@dataclass(frozen=True)
class HitRecord:
    hotspot_id: int
    kind: str
    effector: str
    spawn_t: float
    hit_t: float
    reaction: float
    points: int
    combo: int


@dataclass(frozen=True)
class HitEvent:
    record: HitRecord
    x: float
    y: float


@dataclass(frozen=True)
class MissEvent:
    hotspot: Hotspot


@dataclass(frozen=True)
class StateEvent:
    old: State
    new: State
    t: float


Event = HitEvent | MissEvent | StateEvent


@dataclass
class _Round:
    start: float
    end: float
    next_spawn: float
    pause_start: float | None = None


class Game:
    def __init__(self, config: GameConfig, frame_size: tuple[int, int], rng: random.Random,
                 calibration: Calibration | None = None):
        self.cfg = config
        self.frame_size = frame_size
        self.rng = rng
        self.calibrator = Calibrator(min_vis=config.min_visibility)
        self.calibration: Calibration | None = None
        self.spawner: Spawner | None = None
        self.state = State.CALIBRATING
        self.state_t: float | None = None
        self.feet_enabled = False
        self._pending_cal = calibration
        self._reset_scores()

    # ---- public API -------------------------------------------------------

    def update(self, t: float, body: Body | None, effectors: dict[str, Effector]) -> list[Event]:
        events: list[Event] = []
        if self.state_t is None:
            self.state_t = t
            if self._pending_cal is not None:
                self._set_calibration(self._pending_cal)
                self._enter(State.COUNTDOWN, t, events)
        if body is not None:
            self._last_body_t = t

        if self.state is State.CALIBRATING:
            cal = self.calibrator.update(t, body)
            if cal is not None:
                self._set_calibration(cal)
                self._enter(State.COUNTDOWN, t, events)
        elif self.state is State.COUNTDOWN:
            if t - self.state_t >= self.cfg.countdown:
                self._start_round(t, events)
        elif self.state is State.PLAYING:
            assert self._round is not None
            if self._player_lost(t):
                # Freeze from the moment the player was last seen, so the
                # detection window (pause_after) isn't taken off their clock.
                self._round.pause_start = self._last_body_t if self._last_body_t is not None else t
                self._enter(State.PAUSED, t, events)
            else:
                self._check_hits(t, effectors, events)
                self._expire(t, events)
                if t >= self._round.end:
                    for h in self.active:
                        events.append(MissEvent(h))
                    self.active.clear()
                    self._enter(State.GAME_OVER, t, events)
                else:
                    self._maybe_spawn(t, body)
        elif self.state is State.PAUSED:
            if body is not None:
                self._resume(t, events)

        self._remember(t, effectors)
        return events

    def quick_calibrate(self, t: float, body: Body | None) -> bool:
        if self.state is not State.CALIBRATING:
            return False
        cal = Calibrator.quick(body, self.cfg.min_visibility)
        if cal is None:
            return False
        self._set_calibration(cal)
        self._enter(State.COUNTDOWN, t, [])
        return True

    def restart(self, t: float) -> bool:
        if self.state is not State.GAME_OVER:
            return False
        self._reset_scores()
        self._enter(State.COUNTDOWN, t, [])
        return True

    def countdown_left(self, t: float) -> float:
        return max(0.0, self.cfg.countdown - (t - (self.state_t or t)))

    def time_left(self, t: float) -> float:
        if self._round is None:
            return self.cfg.duration
        if self.state is State.PAUSED and self._round.pause_start is not None:
            t = self._round.pause_start
        return max(0.0, self._round.end - t)

    @property
    def combo_multiplier(self) -> float:
        return 1.0 + self.cfg.combo_step * min(self.combo, self.cfg.combo_cap)

    def summary(self) -> dict:
        rts = [h.reaction * 1000 for h in self.hits]
        total = len(self.hits) + self.misses
        return {
            "score": self.score,
            "hits": len(self.hits),
            "misses": self.misses,
            "accuracy": round(len(self.hits) / total, 3) if total else 0.0,
            "mean_reaction_ms": round(statistics.fmean(rts), 1) if rts else None,
            "median_reaction_ms": round(statistics.median(rts), 1) if rts else None,
            "best_reaction_ms": round(min(rts), 1) if rts else None,
            "best_combo": self.best_combo,
        }

    # ---- internals --------------------------------------------------------

    def _reset_scores(self) -> None:
        self.score = 0
        self.combo = 0
        self.best_combo = 0
        self.misses = 0
        self.hits: list[HitRecord] = []
        self.active: list[Hotspot] = []
        self._round: _Round | None = None
        self._next_id = 0
        self._last_body_t: float | None = None
        self._prev: dict[str, tuple[float, float, float]] = {}
        self._vel: dict[str, tuple[float, float]] = {}
        self._rearm_until: dict[str, float] = {}

    def _enter(self, state: State, t: float, events: list[Event]) -> None:
        events.append(StateEvent(self.state, state, t))
        self.state = state
        self.state_t = t

    def _set_calibration(self, cal: Calibration) -> None:
        self.calibration = cal
        self.spawner = Spawner(cal, self.frame_size, self.rng)
        self.feet_enabled = self.cfg.mode == "hands_feet" and cal.feet_visible

    def _start_round(self, t: float, events: list[Event]) -> None:
        self._round = _Round(start=t, end=t + self.cfg.duration, next_spawn=t)
        self._last_body_t = t  # grace period: don't pause on the very first frame
        self._enter(State.PLAYING, t, events)

    def _resume(self, t: float, events: list[Event]) -> None:
        r = self._round
        assert r is not None and r.pause_start is not None
        shift = t - r.pause_start
        # Freeze the round while paused: push every deadline back by the pause length.
        r.start += shift
        r.end += shift
        r.next_spawn += shift
        for h in self.active:
            h.spawn_t += shift
        r.pause_start = None
        self._prev.clear()  # positions from before the pause must not form a sweep
        self._vel.clear()
        self._enter(State.PLAYING, t, events)

    def _player_lost(self, t: float) -> bool:
        return self._last_body_t is None or t - self._last_body_t >= self.cfg.pause_after

    def _anchor(self, body: Body | None) -> Anchor:
        cal = self.calibration
        assert cal is not None
        if body is not None and body.shoulders_visible(self.cfg.min_visibility):
            cx, cy = body.shoulder_center()
            scale = max(0.6, min(1.6, body.shoulder_width() / cal.shoulder_width))
            return Anchor(cx, cy, scale)
        return Anchor(cal.center[0], cal.center[1], 1.0)

    def _maybe_spawn(self, t: float, body: Body | None) -> None:
        r = self._round
        assert r is not None and self.spawner is not None
        elapsed = t - r.start
        limit = 1 if elapsed < self.cfg.second_hotspot_after else self.cfg.max_active
        if len(self.active) >= limit or t < r.next_spawn:
            return
        progress = min(1.0, elapsed / self.cfg.duration)
        ttl = self.cfg.ttl_start + (self.cfg.ttl_end - self.cfg.ttl_start) * progress
        kind = "foot" if self.feet_enabled and self.rng.random() < self.cfg.foot_probability else "hand"
        anchor = self._anchor(body)
        clearance = self.cfg.spawn_clearance * self.calibration.reach * anchor.scale
        # Wait (up to spawn_max_delay) for the hands to clear an area rather than
        # placing the target somewhere it could be touched by accident.
        strict = t - r.next_spawn < self.cfg.spawn_max_delay
        h = self.spawner.spawn(self._next_id, t, ttl, kind, anchor, self.active,
                               self._predicted_path(kind), clearance, strict)
        if h is None:
            return
        self._next_id += 1
        self.active.append(h)
        r.next_spawn = t + self.rng.uniform(*self.cfg.spawn_gap)

    def _check_hits(self, t: float, effectors: dict[str, Effector], events: list[Event]) -> None:
        cal = self.calibration
        assert cal is not None and self._round is not None
        tol = self.cfg.touch_tolerance * cal.shoulder_width
        max_jump = self.cfg.max_sweep_reach * cal.reach
        for name, e in effectors.items():
            if not self.active:
                return
            if e.visibility < self.cfg.min_visibility or t < self._rearm_until.get(name, -math.inf):
                continue
            kind = effector_kind(name)
            p1 = (e.x, e.y)
            p0 = None
            prev = self._prev.get(name)
            if prev is not None and t - prev[2] <= self.cfg.max_sweep_gap:
                if math.hypot(p1[0] - prev[0], p1[1] - prev[1]) <= max_jump:
                    p0 = (prev[0], prev[1])
            for h in self.active:
                if h.kind == kind and segment_circle_hit(p0, p1, (h.x, h.y), h.radius + tol):
                    self._score_hit(t, h, name, events)
                    self._rearm_until[name] = t + self.cfg.rearm
                    break

    def _score_hit(self, t: float, h: Hotspot, effector: str, events: list[Event]) -> None:
        assert self._round is not None
        reaction = t - h.spawn_t
        speed = max(0.0, 1.0 - reaction / h.ttl)
        points = round((self.cfg.base_points + self.cfg.speed_bonus * speed) * self.combo_multiplier)
        self.combo += 1
        self.best_combo = max(self.best_combo, self.combo)
        self.score += points
        rec = HitRecord(h.id, h.kind, effector, h.spawn_t, t, reaction, points, self.combo)
        self.hits.append(rec)
        self.active.remove(h)
        events.append(HitEvent(rec, h.x, h.y))
        self._round.next_spawn = max(self._round.next_spawn, t + self.rng.uniform(*self.cfg.spawn_gap))

    def _expire(self, t: float, events: list[Event]) -> None:
        assert self._round is not None
        for h in [h for h in self.active if t >= h.expires_at()]:
            self.active.remove(h)
            self.misses += 1
            self.combo = 0
            events.append(MissEvent(h))
            self._round.next_spawn = max(self._round.next_spawn, t + self.rng.uniform(*self.cfg.spawn_gap))

    def _predicted_path(self, kind: str) -> list[tuple[float, float]]:
        """Where each `kind` effector is now and where it will be over the next
        `spawn_lookahead` seconds if it keeps its current velocity. Sampled every
        50 ms. The extrapolation is capped at one reach, so a velocity spike from
        a tracking glitch can't push the "keep clear" zone across the frame."""
        assert self.calibration is not None
        cap = self.calibration.reach
        steps = max(1, round(self.cfg.spawn_lookahead / 0.05))
        pts: list[tuple[float, float]] = []
        for name, (x, y, _) in self._prev.items():
            if effector_kind(name) != kind:
                continue
            vx, vy = self._vel.get(name, (0.0, 0.0))
            speed = math.hypot(vx, vy)
            horizon = self.cfg.spawn_lookahead
            if speed * horizon > cap:
                horizon = cap / speed
            pts += [(x + vx * horizon * k / steps, y + vy * horizon * k / steps) for k in range(steps + 1)]
        return pts

    def _remember(self, t: float, effectors: dict[str, Effector]) -> None:
        for name, e in effectors.items():
            if e.visibility >= self.cfg.min_visibility:
                prev = self._prev.get(name)
                if prev is not None and 0 < t - prev[2] <= self.cfg.max_sweep_gap:
                    dt = t - prev[2]
                    self._vel[name] = ((e.x - prev[0]) / dt, (e.y - prev[1]) / dt)
                else:
                    self._vel[name] = (0.0, 0.0)
                self._prev[name] = (e.x, e.y, t)
            else:
                self._prev.pop(name, None)
                self._vel.pop(name, None)
