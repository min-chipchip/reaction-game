import math
import random

import pytest

from reaction_game.calibration import Calibration
from reaction_game.game import Game, GameConfig, HitEvent, MissEvent, State
from reaction_game.pose import Effector
from reaction_game.synthetic import ScriptedPlayer, make_body

FRAME = (1280, 720)
DT = 1 / 30
CAL = Calibration(center=(640, 220), shoulder_width=120, reach=255, floor_y=622)


class Clock:
    def __init__(self):
        self.t = 0.0

    def tick(self, dt=DT):
        self.t += dt
        return self.t


def drive(game, clock, body_fn, seconds):
    """Step the game with body_fn(t) for `seconds`, using raw effectors (no smoothing)."""
    events = []
    end = clock.t + seconds
    while clock.t < end:
        t = clock.tick()
        body = body_fn(t)
        eff = body.effectors() if body is not None else {}
        events += game.update(t, body, eff)
    return events


def new_game(**cfg):
    return Game(GameConfig(**cfg), FRAME, random.Random(0))


def playing_game(**cfg):
    g = Game(GameConfig(**cfg), FRAME, random.Random(0), calibration=CAL)
    clock = Clock()
    drive(g, clock, lambda t: make_body(), g.cfg.countdown + 0.1)
    assert g.state is State.PLAYING
    return g, clock


def test_calibration_then_countdown_then_playing():
    g, clock = new_game(), Clock()
    drive(g, clock, lambda t: make_body(arms="down"), 1.0)
    assert g.state is State.CALIBRATING
    drive(g, clock, lambda t: make_body(arms="t"), 1.6)
    assert g.state is State.COUNTDOWN
    assert g.calibration.shoulder_width == pytest.approx(120)
    drive(g, clock, lambda t: make_body(), 3.1)
    assert g.state is State.PLAYING
    assert len(g.active) == 1  # first target spawns immediately


def test_quick_calibrate_skips_t_pose():
    g, clock = new_game(), Clock()
    drive(g, clock, lambda t: make_body(), 0.1)
    assert g.quick_calibrate(clock.t, make_body())
    assert g.state is State.COUNTDOWN
    assert g.calibration.reach == pytest.approx(240)


def test_touching_hotspot_scores_with_speed_bonus():
    g, clock = playing_game()
    h = g.active[0]
    events = drive(g, clock, lambda t: make_body(hands={"hand_r": (h.x, h.y)}), DT)
    hits = [e for e in events if isinstance(e, HitEvent)]
    assert len(hits) == 1 and hits[0].record.hotspot_id == h.id
    # touched within one frame of spawning: nearly the full speed bonus
    assert hits[0].record.points >= 195
    assert g.score == hits[0].record.points and g.combo == 1


def test_hand_cannot_hit_foot_target():
    g, clock = playing_game(mode="hands_feet", foot_probability=1.0)
    h = g.active[0]
    assert h.kind == "foot"
    events = drive(g, clock, lambda t: make_body(hands={"hand_r": (h.x, h.y)}), 3 * DT)
    assert not any(isinstance(e, HitEvent) for e in events)
    events = drive(g, clock, lambda t: make_body(feet={"foot_l": (h.x, h.y)}), DT)
    assert any(isinstance(e, HitEvent) for e in events)


def test_fast_swipe_between_frames_still_hits():
    g, clock = playing_game()
    h = g.active[0]
    far = 3 * h.radius
    # frame 1: hand left of target; frame 2: hand right of target. Neither sample is inside.
    left = Effector(h.x - far, h.y, 0.9)
    right = Effector(h.x + far, h.y, 0.9)
    body = make_body()
    assert not g.update(clock.tick(), body, {"hand_r": left})
    events = g.update(clock.tick(), body, {"hand_r": right})
    assert any(isinstance(e, HitEvent) for e in events)


def test_teleport_glitch_does_not_sweep():
    """A tracking jump across the whole frame is a glitch, not a swipe: no hit."""
    g, clock = playing_game()
    h = g.active[0]
    body = make_body()
    a = Effector(0, h.y, 0.9)
    b = Effector(FRAME[0], h.y, 0.9)  # far more than 1.5x reach in one frame
    g.update(clock.tick(), body, {"hand_r": a})
    events = g.update(clock.tick(), body, {"hand_r": b})
    assert not any(isinstance(e, HitEvent) for e in events)


def test_low_visibility_effector_cannot_score():
    g, clock = playing_game()
    h = g.active[0]
    events = g.update(clock.tick(), make_body(), {"hand_r": Effector(h.x, h.y, 0.2)})
    assert not any(isinstance(e, HitEvent) for e in events)


def test_expiry_counts_miss_and_breaks_combo():
    g, clock = playing_game()
    h = g.active[0]
    drive(g, clock, lambda t: make_body(hands={"hand_r": (h.x, h.y)}), DT)
    assert g.combo == 1
    events = drive(g, clock, lambda t: make_body(), g.cfg.ttl_start + 1.0)
    assert any(isinstance(e, MissEvent) for e in events)
    assert g.combo == 0 and g.misses >= 1


def test_pause_freezes_round_and_hotspot_timers():
    g, clock = playing_game()
    h = g.active[0]
    left_before = g.time_left(clock.t)
    spawn_before = h.spawn_t
    drive(g, clock, lambda t: None, 2.0)
    assert g.state is State.PAUSED
    drive(g, clock, lambda t: make_body(), DT)
    assert g.state is State.PLAYING
    assert g.time_left(clock.t) == pytest.approx(left_before, abs=0.1)
    assert h in g.active  # 2 s away didn't burn its lifetime
    assert h.spawn_t > spawn_before + 1.0


def test_round_ends_in_game_over_and_restart_keeps_calibration():
    g, clock = playing_game(duration=5.0)
    drive(g, clock, lambda t: make_body(), 5.1)
    assert g.state is State.GAME_OVER and not g.active
    assert g.restart(clock.t)
    assert g.state is State.COUNTDOWN and g.score == 0 and g.calibration is CAL


def test_scripted_player_full_round():
    g, clock = new_game(duration=20.0), Clock()
    bot = ScriptedPlayer(reaction=0.4)
    drive(g, clock, lambda t: bot.body(t, g), 1.7 + 3.1 + 20.5)
    assert g.state is State.GAME_OVER
    s = g.summary()
    assert s["hits"] >= 10 and s["misses"] == 0
    assert 400 <= s["median_reaction_ms"] <= 500
    assert s["best_combo"] == s["hits"]


def until_first_target(game, clock, body_fn, limit=5.0):
    """Drive until the first target appears (spawning may wait for the hands to clear an area)."""
    end = clock.t + limit
    while not game.active and clock.t < end:
        drive(game, clock, body_fn, DT)
    assert game.active, "no target spawned"
    return game.active[0]


def test_target_never_spawns_near_a_still_hand():
    """Regression for the live-session bug: targets appearing under a hand that was already there."""
    held = {"hand_r": (760, 120), "hand_l": (520, 200)}  # arms up, inside the target arc
    for seed in range(30):
        g = Game(GameConfig(), FRAME, random.Random(seed), calibration=CAL)
        h = until_first_target(g, Clock(), lambda t: make_body(hands=held))
        for x, y in held.values():
            assert math.hypot(h.x - x, h.y - y) >= g.cfg.spawn_clearance * CAL.reach


def test_target_avoids_where_a_moving_hand_is_heading():
    vx = 900.0  # px/s, sweeping right at head height, starting just before play begins
    for seed in range(30):
        g = Game(GameConfig(), FRAME, random.Random(seed), calibration=CAL)
        t_move = g.cfg.countdown - 0.3
        pos = lambda t: (200 + vx * max(0.0, t - t_move), 120)
        h = until_first_target(g, Clock(), lambda t: make_body(hands={"hand_r": pos(t)}))
        x0, y0 = pos(h.spawn_t)
        for k in range(6):  # where the hand will be over the next 0.25 s
            x = x0 + vx * 0.05 * k
            assert math.hypot(h.x - x, h.y - y0) >= g.cfg.spawn_clearance * CAL.reach - 1e-6


def test_waits_for_space_then_spawns_anyway():
    """If no spot is clear, the spawn waits, but never longer than spawn_max_delay."""
    g = Game(GameConfig(spawn_clearance=5.0), FRAME, random.Random(0), calibration=CAL)
    h = until_first_target(g, Clock(), lambda t: make_body())
    waited = h.spawn_t - g._round.start
    assert g.cfg.spawn_max_delay - DT <= waited <= g.cfg.spawn_max_delay + DT


def test_aimless_sweeping_gets_almost_no_impossible_hits():
    """Hands that never aim, always moving fast: hits under 150 ms (faster than human reaction) must be rare."""
    fast = total = 0
    for seed in range(5):
        g = Game(GameConfig(duration=30), FRAME, random.Random(seed), calibration=CAL)
        clock = Clock()
        d = 0.75 * CAL.reach

        def body(t):
            a1, a2 = 2 * math.pi * 0.7 * t, -2 * math.pi * 0.55 * t + 1.0
            return make_body(hands={"hand_r": (640 + d * math.cos(a1), 220 - d * abs(math.sin(a1))),
                                    "hand_l": (640 + d * math.cos(a2), 220 - d * abs(math.sin(a2)))})

        for e in drive(g, clock, body, 34):
            if isinstance(e, HitEvent):
                total += 1
                fast += e.record.reaction < 0.15
    assert total > 50
    assert fast / total < 0.03  # was 29% before spawn clearance + lookahead
