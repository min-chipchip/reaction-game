import random
import statistics

from reaction_game.filters import EffectorSmoother, OneEuroFilter
from reaction_game.pose import Effector

FPS = 30.0


def test_reduces_jitter_at_rest():
    rng = random.Random(1)
    f = OneEuroFilter(min_cutoff=1.5, beta=0.01)
    raw, out = [], []
    for i in range(300):
        x = 500 + rng.gauss(0, 3)  # ~3 px landmark jitter
        raw.append(x)
        out.append(f(x, i / FPS))
    # 1.5 Hz cutoff at 30 FPS -> theoretical ~0.37x residual noise
    assert statistics.pstdev(out[30:]) < 0.5 * statistics.pstdev(raw[30:])


def test_tracks_fast_motion_with_little_lag():
    """A 1500 px/s swipe: the adaptive cutoff should keep lag small."""
    f = OneEuroFilter(min_cutoff=1.5, beta=0.01)
    slow = OneEuroFilter(min_cutoff=1.5, beta=0.0)  # plain low-pass, same rest cutoff
    for i in range(30):
        f(100.0, i / FPS)
        slow(100.0, i / FPS)
    x = 100.0
    for i in range(30, 40):
        x += 1500 / FPS
        y_adapt, y_fixed = f(x, i / FPS), slow(x, i / FPS)
    assert x - y_adapt < 0.5 * (x - y_fixed)
    assert x - y_adapt < 120  # under ~2.5 frames of motion


def test_duplicate_timestamp_is_ignored():
    f = OneEuroFilter()
    assert f(10.0, 1.0) == 10.0
    assert f(99.0, 1.0) == 10.0


def test_smoother_passes_low_visibility_through_and_resets_after_gap():
    s = EffectorSmoother(reset_after=0.3)
    s.update(0.0, {"hand_l": Effector(100, 100, 0.9)})
    s.update(1 / FPS, {"hand_l": Effector(100, 100, 0.9)})
    occluded = s.update(2 / FPS, {"hand_l": Effector(900, 900, 0.1)})["hand_l"]
    assert (occluded.x, occluded.y) == (900, 900)  # raw, untouched
    # Reappears far away after a long gap: no slow glide from the old position.
    back = s.update(1.0, {"hand_l": Effector(700, 300, 0.9)})["hand_l"]
    assert (back.x, back.y) == (700, 300)
