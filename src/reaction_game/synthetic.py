"""Synthetic bodies and a scripted player, for tests and camera-free demos."""
from __future__ import annotations

from .pose import (L_ANKLE, L_ELBOW, L_FOOT, L_HEEL, L_HIP, L_INDEX, L_KNEE, L_SHOULDER, L_WRIST, NOSE,
                   NUM_LANDMARKS, R_ANKLE, R_ELBOW, R_FOOT, R_HEEL, R_HIP, R_INDEX, R_KNEE, R_SHOULDER,
                   R_WRIST, Body)

HAND_LMS = {"hand_l": (L_WRIST, L_INDEX), "hand_r": (R_WRIST, R_INDEX)}
FOOT_LMS = {"foot_l": (L_ANKLE, L_FOOT), "foot_r": (R_ANKLE, R_FOOT)}


def make_body(cx: float = 640, cy: float = 220, sw: float = 120, arms: str = "down",
              vis: float = 0.95, hands: dict[str, tuple[float, float]] | None = None,
              feet: dict[str, tuple[float, float]] | None = None) -> Body:
    """A standing figure facing the camera with the shoulder centre at (cx, cy).

    `hands` and `feet` place an effector (the midpoint of its two landmarks)
    exactly at the given point.
    """
    lm = [(cx, cy, vis)] * NUM_LANDMARKS
    lm = list(lm)

    def put(i, x, y):
        lm[i] = (x, y, vis)

    put(NOSE, cx, cy - 0.9 * sw)
    for side, (sh, el, wr, ix, hp, kn, an, he, ft) in {
        +1: (L_SHOULDER, L_ELBOW, L_WRIST, L_INDEX, L_HIP, L_KNEE, L_ANKLE, L_HEEL, L_FOOT),
        -1: (R_SHOULDER, R_ELBOW, R_WRIST, R_INDEX, R_HIP, R_KNEE, R_ANKLE, R_HEEL, R_FOOT),
    }.items():
        put(sh, cx + side * sw / 2, cy)
        if arms == "t":
            put(el, cx + side * 1.25 * sw, cy)
            put(wr, cx + side * 2.0 * sw, cy)
            put(ix, cx + side * 2.25 * sw, cy)
        else:
            put(el, cx + side * sw / 2, cy + 0.75 * sw)
            put(wr, cx + side * sw / 2, cy + 1.5 * sw)
            put(ix, cx + side * sw / 2, cy + 1.75 * sw)
        put(hp, cx + side * 0.35 * sw, cy + 1.5 * sw)
        put(kn, cx + side * 0.35 * sw, cy + 2.4 * sw)
        put(an, cx + side * 0.35 * sw, cy + 3.3 * sw)
        put(he, cx + side * 0.35 * sw, cy + 3.35 * sw)
        put(ft, cx + side * 0.45 * sw, cy + 3.4 * sw)

    for table, targets in ((HAND_LMS, hands), (FOOT_LMS, feet)):
        for name, (x, y) in (targets or {}).items():
            a, b = table[name]
            put(a, x, y - 0.1 * sw)
            put(b, x, y + 0.1 * sw)
    return Body(lm)


class ScriptedPlayer:
    """A bot that reaches for the oldest active hotspot after `reaction` seconds.

    It moves its hand there in a single frame, so it also exercises the swept
    hit test.
    """

    def __init__(self, reaction: float = 0.4, **body_kw):
        self.reaction = reaction
        self.body_kw = body_kw

    def body(self, t: float, game) -> Body:
        from .game import State

        if game.state is State.CALIBRATING:
            return make_body(arms="t", **self.body_kw)
        target = min(game.active, key=lambda h: h.spawn_t, default=None)
        if target is None or t - target.spawn_t < self.reaction:
            return make_body(**self.body_kw)
        if target.kind == "foot":
            return make_body(feet={"foot_r": (target.x, target.y)}, **self.body_kw)
        return make_body(hands={"hand_r": (target.x, target.y)}, **self.body_kw)
