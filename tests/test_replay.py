from reaction_game.game import GameConfig, State
from reaction_game.replay import Recorder, replay_headless
from reaction_game.session import Session, SessionConfig
from reaction_game.synthetic import ScriptedPlayer

FRAME = (1280, 720)


def play_and_record(path, seed=7, keys=()):
    cfg = SessionConfig(game=GameConfig(duration=15.0, mode="hands_feet"), seed=seed)
    session = Session(cfg, FRAME)
    rec = Recorder(path, cfg, FRAME)
    bot = ScriptedPlayer(reaction=0.35)
    t = 100.0
    keys = dict(keys)
    for i in range(int(22 / (1 / 30))):
        t += 1 / 30
        # drop a few frames to mimic tracking loss mid-round
        body = None if 400 <= i < 430 else bot.body(t, session.game)
        session.step(t, body)
        rec.frame(t, body)
        if i in keys:
            session.key(t, keys[i])
            rec.key(t, keys[i])
    rec.close()
    return session


def test_replay_reproduces_live_session_exactly(tmp_path):
    path = tmp_path / "s.jsonl"
    live = play_and_record(path)
    assert live.game.state is State.GAME_OVER
    assert live.game.summary()["hits"] > 5

    replayed = replay_headless(path)
    assert replayed.game.summary() == live.game.summary()
    assert replayed.game.hits == live.game.hits


def test_replay_reproduces_keys(tmp_path):
    path = tmp_path / "k.jsonl"
    # 's' on frame 5 skips the T-pose (quick calibration)
    live = play_and_record(path, keys={5: "s"})
    replayed = replay_headless(path)
    assert replayed.game.calibration == live.game.calibration
    assert replayed.game.summary() == live.game.summary()
