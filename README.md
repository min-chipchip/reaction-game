# Reaction Game

A webcam motion drill, like the light-up reaction pads used in basketball training:
- Targets appear around you on screen.
- You touch them with your hands (and optionally your feet), tracked by your body's pose.
- Faster touches score more.

Built with Python, OpenCV and MediaPipe Pose (BlazePose, on-device). See [docs/DESIGN.md](docs/DESIGN.md) for the pipeline, the latency budget, and the trade-offs.

## Setup

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
curl -L -o models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

(Optional) For `--model full`, download `pose_landmarker_full.task` from the same URL with `lite` replaced by `full`.

On macOS, the first run asks for camera permission for your terminal app.

## Play

```bash
.venv/bin/reaction-game --debug
```

1. Stand back until your whole body is visible (2–3 m from the camera works best).
2. **Hold a T-pose** for 1.5 s to calibrate. Press `S` to skip, but reach is then only estimated.
3. After the 3-2-1 countdown, touch the **HAND** targets (and **FOOT** targets with `--mode hands_feet`) before their ring runs out.

| Key | Action |
|-----|--------|
| `D` | toggle the debug overlay (FPS, per-stage latency, raw vs smoothed effectors) |
| `S` | skip T-pose calibration |
| `R` | play again (after game over) |
| `Q` / `Esc` | quit |

Scoring:
- Each hit is worth `(100 + up to 100 speed bonus) × combo multiplier`.
- The combo multiplier grows by +0.1 per consecutive hit, up to ×2.0. A missed target resets it.
- Targets live for 3.0 s at the start of the round, shrinking to 1.5 s by the end.
- A second simultaneous target starts appearing after 15 s.
- When the round ends, the results are written to `results/session_*.csv`.

## Useful flags

```text
--model lite|full|heavy   accuracy vs latency
--mode hands|hands_feet   foot targets need your feet visible during calibration
--duration 60             round length (s)
--seed N                  reproducible target layout
--record run.jsonl        save landmarks + key presses
--replay run.jsonl        replay a recording (add --headless to print the result only)
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

The game logic is driven by an injected clock and synthetic bodies (`reaction_game/synthetic.py`), so the tests need no camera or model. They cover:
- the One Euro filter
- swept collision, including fast swipes and teleport glitches
- spawning within reach, scaling with player distance, and never spawning on a limb that is already there
- calibration
- every state transition, including pause and resume
- a full bot round
- record → replay reproducing the exact score

## Layout

```
src/reaction_game/
  main.py         CLI, live loop, replay viewer
  capture.py      threaded camera, newest-frame-only
  pose.py         MediaPipe PoseLandmarker wrapper -> Body / Effector
  filters.py      One Euro filter + per-effector smoother
  collision.py    swept segment-vs-circle hit test
  calibration.py  T-pose calibration
  hotspots.py     Hotspot + body-relative Spawner
  game.py         state machine and scoring
  session.py      deterministic core shared by live and replay
  replay.py       JSONL record/replay
  render.py       OpenCV drawing
  metrics.py      per-stage latency, FPS
  synthetic.py    synthetic bodies + scripted player
```
