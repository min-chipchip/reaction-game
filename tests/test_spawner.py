import math
import random

from reaction_game.calibration import Calibration, Calibrator, is_t_pose
from reaction_game.hotspots import HAND_DIST_RANGE, Anchor, Spawner
from reaction_game.synthetic import make_body

FRAME = (1280, 720)
CAL = Calibration(center=(640, 220), shoulder_width=120, reach=255, floor_y=622)


def test_hand_targets_in_frame_and_within_reach():
    sp = Spawner(CAL, FRAME, random.Random(0))
    anchor = Anchor(640, 220, 1.0)
    lo, hi = HAND_DIST_RANGE
    for i in range(500):
        h = sp.spawn(i, 0.0, 2.0, "hand", anchor, [])
        assert h.radius <= h.x <= FRAME[0] - h.radius and h.radius <= h.y <= FRAME[1] - h.radius
        d = math.hypot(h.x - anchor.x, h.y - anchor.y)
        assert lo * CAL.reach - 1e-6 <= d <= hi * CAL.reach + 1e-6


def test_targets_scale_with_player_distance():
    sp = Spawner(CAL, FRAME, random.Random(0))
    near = Anchor(640, 300, 1.5)  # player stepped closer: looks 1.5x bigger
    for i in range(200):
        h = sp.spawn(i, 0.0, 2.0, "hand", near, [])
        assert math.isclose(h.radius, CAL.hotspot_radius * 1.5)
        assert math.hypot(h.x - near.x, h.y - near.y) <= HAND_DIST_RANGE[1] * CAL.reach * 1.5 + 1e-6


def test_no_overlap_with_existing():
    sp = Spawner(CAL, FRAME, random.Random(1))
    anchor = Anchor(640, 220, 1.0)
    for i in range(200):
        a = sp.spawn(0, 0.0, 2.0, "hand", anchor, [])
        b = sp.spawn(1, 0.0, 2.0, "hand", anchor, [a])
        assert math.hypot(a.x - b.x, a.y - b.y) >= a.radius + b.radius


def test_foot_targets_on_floor_line():
    sp = Spawner(CAL, FRAME, random.Random(2))
    for i in range(100):
        h = sp.spawn(i, 0.0, 2.0, "foot", Anchor(640, 220, 1.0), [])
        assert abs(h.y - (CAL.floor_y - 0.3 * h.radius)) < 1e-6
        assert abs(h.x - 640) >= 0.8 * CAL.shoulder_width


def test_calibrator_measures_t_pose():
    assert is_t_pose(make_body(arms="t"))
    assert not is_t_pose(make_body(arms="down"))
    c = Calibrator(hold_seconds=1.0)
    cal = None
    t = 0.0
    while cal is None and t < 3:
        cal = c.update(t, make_body(arms="t"))
        t += 1 / 30
    assert cal is not None and 0.95 <= t <= 1.1
    assert math.isclose(cal.shoulder_width, 120)
    assert math.isclose(cal.reach, 2.125 * 120)
    assert cal.feet_visible


def test_calibrator_resets_when_pose_breaks():
    c = Calibrator(hold_seconds=1.0)
    for i in range(20):
        c.update(i / 30, make_body(arms="t"))
    assert c.progress(19 / 30) > 0.5
    c.update(20 / 30, make_body(arms="down"))
    assert c.progress(20 / 30) == 0.0


def test_keeps_clearance_from_hands_and_feet():
    sp = Spawner(CAL, FRAME, random.Random(3))
    clearance = 0.6 * CAL.reach
    resting = {"hand": [(580, 415), (700, 415)], "foot": [(592, 622), (688, 622)]}
    for i in range(300):
        for kind, pts in resting.items():
            h = sp.spawn(i, 0.0, 2.0, kind, Anchor(640, 220, 1.0), [], pts, clearance)
            assert all(math.hypot(h.x - x, h.y - y) >= clearance for x, y in pts)


def test_strict_returns_none_when_no_spot_is_clear():
    sp = Spawner(CAL, FRAME, random.Random(4))
    everywhere = [(x, y) for x in range(0, 1280, 40) for y in range(0, 720, 40)]
    assert sp.spawn(0, 0.0, 2.0, "hand", Anchor(640, 220, 1.0), [], everywhere, 100, strict=True) is None
    # non-strict still places one (the least-bad spot)
    assert sp.spawn(0, 0.0, 2.0, "hand", Anchor(640, 220, 1.0), [], everywhere, 100) is not None
