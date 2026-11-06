"""Session: the deterministic core shared by the live loop and replay.

    Body (raw) -> EffectorSmoother -> Game.update -> events

Everything downstream of pose inference goes through `Session.step` and
`Session.key`. A recording of (timestamp, Body, key) plus the seed therefore
reproduces a live game exactly.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from .filters import EffectorSmoother
from .game import Event, Game, GameConfig
from .pose import Body, Effector

GAME_KEYS = {"s", "r"}  # keys that change game state, so they are recorded


@dataclass(frozen=True)
class SessionConfig:
    game: GameConfig = field(default_factory=GameConfig)
    seed: int = 0
    smooth_min_cutoff: float = 1.5
    smooth_beta: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SessionConfig":
        g = dict(d["game"])
        g["spawn_gap"] = tuple(g["spawn_gap"])
        return cls(game=GameConfig(**g), seed=d["seed"],
                   smooth_min_cutoff=d["smooth_min_cutoff"], smooth_beta=d["smooth_beta"])


class Session:
    def __init__(self, config: SessionConfig, frame_size: tuple[int, int]):
        self.config = config
        self.frame_size = frame_size
        self.rng = random.Random(config.seed)
        self.smoother = EffectorSmoother(config.smooth_min_cutoff, config.smooth_beta,
                                         min_visibility=config.game.min_visibility)
        self.game = Game(config.game, frame_size, self.rng)
        self.body: Body | None = None
        self.raw_effectors: dict[str, Effector] = {}
        self.effectors: dict[str, Effector] = {}

    def step(self, t: float, body: Body | None) -> list[Event]:
        self.body = body
        self.raw_effectors = body.effectors() if body is not None else {}
        self.effectors = self.smoother.update(t, self.raw_effectors)
        return self.game.update(t, body, self.effectors)

    def key(self, t: float, key: str) -> bool:
        if key == "s":
            return self.game.quick_calibrate(t, self.body)
        if key == "r":
            return self.game.restart(t)
        return False
