"""CLI entry point: live webcam game, or replay of a recording."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from .game import GameConfig, State, StateEvent
from .metrics import Metrics
from .pose import MODEL_FILES, MODEL_URL, model_path
from .replay import Recorder, apply, load, replay_headless, session_from_header
from .session import GAME_KEYS, Session, SessionConfig

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
WINDOW = "Reaction Game"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Touch the hotspots with your hands (and feet).")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--model", choices=sorted(MODEL_FILES), default="lite")
    p.add_argument("--mode", choices=["hands", "hands_feet"], default="hands")
    p.add_argument("--duration", type=float, default=60.0, help="round length in seconds")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (default: random)")
    p.add_argument("--debug", action="store_true", help="show latency/FPS overlay (toggle with D)")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--record", type=Path, help="write landmarks + keys to this JSONL file")
    p.add_argument("--replay", type=Path, help="replay a recording instead of using the camera")
    p.add_argument("--headless", action="store_true", help="with --replay: print the result, no window")
    return p.parse_args(argv)


def write_results(session: Session, out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"session_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hotspot_id", "kind", "effector", "reaction_ms", "points", "combo"])
        for h in session.game.hits:
            w.writerow([h.hotspot_id, h.kind, h.effector, round(h.reaction * 1000, 1), h.points, h.combo])
    return path


def _key_name(code: int) -> str | None:
    if code in (-1, 255):
        return None
    if code == 27:
        return "esc"
    return chr(code).lower() if 32 <= code < 127 else None


def run_live(args) -> int:
    import cv2

    from .capture import CameraError, CameraStream
    from .pose import PoseTracker
    from .render import Renderer

    mfile = model_path(args.model, MODELS_DIR)
    if not mfile.exists():
        url = MODEL_URL.format(variant=args.model)
        print(f"Model not found: {mfile}\nDownload it with:\n  curl -L -o {mfile} {url}", file=sys.stderr)
        return 2

    try:
        cam = CameraStream(args.camera, args.width, args.height).start()
    except CameraError as e:
        print(e, file=sys.stderr)
        return 1
    frame_size = cam.frame_size
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    cfg = SessionConfig(game=GameConfig(duration=args.duration, mode=args.mode), seed=seed)
    session = Session(cfg, frame_size)
    tracker = PoseTracker(mfile)
    recorder = Recorder(args.record, cfg, frame_size) if args.record else None
    metrics = Metrics()
    renderer = Renderer()
    debug = args.debug

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print(f"camera {frame_size[0]}x{frame_size[1]}, model={args.model}, mode={args.mode}, seed={seed}")

    try:
        while True:
            got = cam.read()
            if got is None:
                print("camera stalled", file=sys.stderr)
                break
            frame, ts, _ = got
            with metrics.stage("flip"):
                frame = cv2.flip(frame, 1)  # mirror: moving your right hand moves the right-side dot
            with metrics.stage("pose"):
                body = tracker.detect(frame, ts)
            with metrics.stage("game"):
                events = session.step(ts, body)
            if recorder:
                recorder.frame(ts, body)
            renderer.on_events(ts, events)
            with metrics.stage("render"):
                extra = [f"model {args.model}  dropped {cam.dropped}"]
                renderer.draw(frame, session, ts, metrics, debug, extra)
            with metrics.stage("display"):
                cv2.imshow(WINDOW, frame)
                key = _key_name(cv2.waitKey(1))
            metrics.add("e2e", (time.monotonic() - ts) * 1000.0)
            metrics.tick()

            for e in events:
                if isinstance(e, StateEvent) and e.new is State.GAME_OVER:
                    path = write_results(session)
                    print(json.dumps(session.game.summary()), f"-> {path}")
            if key in ("q", "esc"):
                break
            if key == "d":
                debug = not debug
            elif key in GAME_KEYS:
                session.key(ts, key)
                if recorder:
                    recorder.key(ts, key)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.stop()
        tracker.close()
        if recorder:
            recorder.close()
            print(f"recorded -> {args.record}")
        cv2.destroyAllWindows()
    print("latency (ms):", ", ".join(f"{n} p50={a:.1f} p95={b:.1f}" for n, a, b in metrics.report()))
    print(f"fps={metrics.fps:.1f}  dropped_frames={cam.dropped}")
    return 0


def run_replay(args) -> int:
    if args.headless:
        session = replay_headless(args.replay)
        print(json.dumps({"state": session.game.state.value, **session.game.summary()}))
        return 0

    import cv2
    import numpy as np

    from .render import Renderer

    header, records = load(args.replay)
    session = session_from_header(header)
    w, h = session.frame_size
    renderer = Renderer()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    wall0 = t0 = None
    for rec in records:
        events = apply(session, rec)
        if rec["type"] != "frame":
            continue
        t = rec["t"]
        if t0 is None:
            wall0, t0 = time.monotonic(), t
        renderer.on_events(t, events)
        img = np.full((h, w, 3), 30, np.uint8)
        renderer.draw(img, session, t, None, args.debug)
        cv2.imshow(WINDOW, img)
        delay = (t - t0) - (time.monotonic() - wall0)  # play back at recorded speed
        if _key_name(cv2.waitKey(max(1, int(delay * 1000)))) in ("q", "esc"):
            break
    print(json.dumps(session.game.summary()))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    return run_replay(args) if args.replay else run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
