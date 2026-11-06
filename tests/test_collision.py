from reaction_game.collision import segment_circle_hit


def test_point_inside_and_outside():
    assert segment_circle_hit(None, (105, 100), (100, 100), 10)
    assert not segment_circle_hit(None, (120, 100), (100, 100), 10)


def test_fast_swipe_through_circle_hits():
    # Both samples lie outside the circle, but the hand passed through it.
    assert segment_circle_hit((0, 100), (200, 100), (100, 100), 10)


def test_segment_stopping_short_misses():
    assert not segment_circle_hit((0, 100), (80, 100), (100, 100), 10)


def test_segment_passing_beside_misses():
    assert not segment_circle_hit((0, 130), (200, 130), (100, 100), 10)


def test_zero_length_segment_is_point_test():
    assert segment_circle_hit((100, 100), (100, 100), (100, 100), 1)
