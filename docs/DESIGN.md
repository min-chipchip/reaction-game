# Design notes

These notes are written to be talked through. Each section covers one decision, why it was made, and what it costs.

## 1. Pipeline

```
 camera thread               main thread (one iteration per frame)
 ┌──────────────┐  newest   ┌──────┐   ┌──────────┐   ┌─────────┐   ┌──────┐   ┌────────┐   ┌─────────┐
 │ cap.read()   │──frame───▶│ flip │──▶│ pose     │──▶│ One Euro│──▶│ game │──▶│ render │──▶│ display │
 │ ts=monotonic │  only     └──────┘   │ (VIDEO)  │   │ smooth  │   │ FSM  │   └────────┘   └─────────┘
 └──────────────┘                      └──────────┘   └─────────┘   └──────┘
                                         Body           Effectors     Events
                                       (33 lm, px)    (2 hands,      (hit/miss/
                                                       2 feet)        state)
```

- **Newest frame only.** Reading frames from the driver queue in the main loop means any slow frame adds permanent lag, because the queue fills and every later frame waits behind it. Instead, the capture thread overwrites a single slot, and the main loop takes whatever is newest. Skipped frames are counted in `dropped` on the debug HUD. The rule: **drop frames, never queue them.**
- **The game state depends only on frame timestamps.** Every stage after pose (smoothing, game, scoring) uses the frame's capture timestamp and never reads the wall clock. This is what makes `Session` deterministic and replays exact.
- **VIDEO running mode.** VIDEO mode lets MediaPipe follow the previous pose and skip the person detector on most frames. That is cheaper and steadier than IMAGE mode, but it requires strictly increasing timestamps, which `PoseTracker` enforces.
- **Mirroring before inference.** The frame is flipped before pose runs, so landmarks come out already in display coordinates and no coordinate transform is needed later. Left and right labels are never used, which keeps the mirroring safe. Effectors are matched by *kind* (hand or foot), not side, and the T-pose check ignores sides.

## 2. Latency budget

Target: at least 25 FPS, and under ~60 ms from capture to `imshow` with the lite model.

Measured 2026-11-06 on an Apple Silicon Mac, built-in camera at 1280×720 and 30 FPS, CPU (XNNPACK), one 60 s round:

| stage | what | lite p50 / p95 (ms) | full p50 / p95 (ms) |
|---|---|---|---|
| flip | `cv2.flip` 1280×720 | 0.1 / 0.3 | _todo_ |
| pose | BGR→RGB + PoseLandmarker | 11.3 / 11.9 | _todo_ |
| game | smoothing + hit tests + spawn | 0.0 / 0.0 | |
| render | overlays + alpha blends | 2.6 / 2.9 | |
| display | `imshow` + `waitKey(1)` | **16.0 / 16.1** | |
| **e2e** | capture timestamp → frame shown | **30.1 / 30.8** | _todo_ |

FPS 30.0 (camera-limited). 11 frames dropped out of about 1,950 (0.6%).

**Reading it:**
- **Display is the biggest stage, not inference.** `imshow` + `waitKey` blocks for about one 60 Hz refresh (16 ms), because the macOS window is presented at the display's refresh rate.
- **The loop has almost no slack.** One iteration takes ~30 ms against a 33.3 ms frame budget. That is why the few dropped frames appear: any hiccup over ~3 ms costs a camera frame.
- **The next win is outside the model.** Presenting frames from a separate thread, or with a GPU-backed window, would remove the 16 ms block and leave room for the full model.

To re-measure, run `reaction-game --debug` (add `--model full`). A p50/p95 summary is printed on exit.

**What `e2e` does not include:**
- The camera's own latency (exposure, USB transfer, driver buffering): typically 30–60 ms on a webcam.
- Display scan-out: about one refresh, ~8–16 ms.

A true glass-to-glass number needs an external measurement, for example filming an LED and the screen with a high-speed camera.

**Where the time goes and what to do about it:**
- Pose inference dominates the budget.
- Options to reduce it: the lite model, a smaller input (the model resizes to 256×256 internally anyway, so capturing at 640×480 mostly saves the color conversion and flip), and a GPU delegate.
- On an embedded console you would also pin threads and avoid copies between capture, inference and render.

## 3. Smoothing: One Euro filter

Raw landmarks jitter by a few pixels at rest. A fixed low-pass filter forces a choice:
- Strong smoothing gives a steady cursor that lags behind fast reaches.
- Weak smoothing gives a responsive cursor that buzzes.

The One Euro filter sets its cutoff from the speed of the signal (`min_cutoff + beta·|speed|`), so it is smooth when still and responsive when moving.

- `min_cutoff = 1.5 Hz` cuts jitter at rest to ~0.37× (test: `test_reduces_jitter_at_rest`).
- `beta = 0.01` (px units). In a 1500 px/s swipe, lag is less than half that of the same filter with `beta = 0` (`test_tracks_fast_motion_with_little_lag`).
- **Cost:** with the scripted bot, which teleports its hand onto the target, hits register about 60 ms after the hand arrives (500 ms scripted reaction → ~560 ms measured). Real hands move continuously, so the real cost is lower, but it is not zero. You can see it live: the debug overlay draws the raw (red) and smoothed (green) points.
- Occluded samples (visibility < 0.5) are not fed into the filter. After 0.3 s unseen, the filter resets, so a hand that reappears elsewhere jumps there instead of gliding from where it was lost.

## 4. Hit detection

- **Swept test.** At 30 FPS a fast swipe moves 150+ px per frame, so a target can sit between two samples. The game tests the *segment* from the previous position to the current one against the circle (`collision.segment_circle_hit`).
- **The swept test has guards**, because a segment can also be wrong:
  - It only applies if the previous sample is under 150 ms old. After a gap there is no meaningful path between the two samples.
  - It only applies if the jump is under 1.5× reach. Larger jumps are tracking glitches, such as the model snapping to another limb or person, not motion. Test: `test_teleport_glitch_does_not_sweep`.
- **Visibility gate.** An effector below 0.5 visibility can't score. MediaPipe still guesses positions for occluded limbs, and those guesses shouldn't earn points.
- **Re-arm.** The effector that just scored can't score again for 200 ms, so one sweep can't clear two overlapping targets.
- **Kind matching.** Hands hit HAND targets and feet hit FOOT targets.
- **Touch tolerance.** The hit radius is the target radius plus 0.08 shoulder widths, because a hand is not a point.

## 5. Calibration and spawning relative to the body

Pixels are the wrong unit. A child 3 m away and an adult 1.5 m away differ by more than 2× in pixel size. So:

- **T-pose calibration.** A 1.5 s hold measures shoulder width, reach (shoulder centre to palm), and floor line, using medians over all frames in the hold. The check ignores left/right on purpose, and the hold restarts if the pose breaks.
- **Hand targets** are placed on an arc from −15° to 195° around the *current* shoulder centre, at 0.5–0.95 × reach. Every target needs a real reach, and none is out of reach.
- **Scale tracking.** Reach is rescaled each spawn by `current shoulder width / calibrated shoulder width`, clamped to 0.6–1.6. A player who steps closer gets bigger, farther targets.
- **Foot targets** sit on the floor line, 1.5–2.5 shoulder widths to either side. Foot mode only turns on if both feet were visible during calibration.
- **No free points: keep clear of the hands now, and of where they are heading.**
  - **First version:** targets only avoided a 2r circle around each effector's current position. A test caught a foot target spawning under the standing foot.
  - **The first live session showed this wasn't enough.** 8 of 67 hits had reaction times under 200 ms, two of them one frame (33 ms). No human reacts that fast. The CSV histogram had a clear gap between 200 and 300 ms separating these from the real reactions. A hand moving at ~1000 px/s crosses a 2r (~100 px) gap in about 100 ms, so targets were appearing where a hand was about to be.
  - **The fix has three parts:**
    1. The clearance becomes body-relative: 0.6 × reach (~150 px at a typical distance).
    2. It is measured against each same-kind effector's *predicted path*, extrapolated linearly from its current velocity over the next 0.3 s. The extrapolation is capped at one reach so a velocity spike can't block the whole frame.
    3. If no candidate spot is clear, the spawn **waits** a frame and tries again. After 0.5 s it takes the spot with the most clearance, so a player who never stops moving can't stall the game.
  - **Result**, measured with a bot whose hands sweep fast and never aim: hits under 150 ms went from **29% to under 1%** (`test_aimless_sweeping_gets_almost_no_impossible_hits`). The remainder come from the bot's unnatural sharp bounce at shoulder height, which a straight-line prediction can't follow.
  - **Possible follow-up: a minimum reaction time.** Hits under ~150 ms wouldn't count, which would guard against free points whatever their cause. It wasn't needed to fix this bug.

## 6. Game state machine

```
CALIBRATING ──T-pose held / S──▶ COUNTDOWN ──3 s──▶ PLAYING ──time up──▶ GAME_OVER
                                     ▲                 │  ▲                 │
                                     │       no body ≥ 0.75 s │ body back   │
                                     │                 ▼  │                 │
                                     │               PAUSED                 │
                                     └──────────────────R───────────────────┘
```

- **Pausing.** The pause freezes the round clock, all target timers and the spawn schedule *from the last frame the body was seen*, so the 0.75 s it takes to detect the loss isn't charged to the player. On resume, every deadline moves later by the pause length, and stored positions are cleared so no sweep spans the pause.
- **Difficulty ramp.** Target lifetime shrinks from 3.0 s to 1.5 s over the round, and a second simultaneous target unlocks after 15 s.

## 7. Testability and replay

- `Game` takes `t` as an argument and never reads a clock. `Session` is the single path from Body to events, used by both the live loop and replay.
- Recordings store full-precision landmarks, keys and the seed. `test_replay_reproduces_live_session_exactly` asserts the hit lists are identical, including through a mid-round tracking dropout.
- **Why replay matters for this team:**
  - Bug reports become reproducible: "the jump didn't count" turns into a file you can replay.
  - Filter and threshold changes can be regression-tested against a library of real movement.
  - Model upgrades can be compared on the same input, by re-running pose on recorded *video*. That is the next step, since this records landmarks only.

## 8. Known failure modes

| Problem | Effect | Mitigation here / next step |
|---|---|---|
| Low light, motion blur | landmark confidence drops; fast limbs blur out | visibility gate and swept test / exposure control, higher FPS camera |
| Self-occlusion (arm across the body) | wrong limb or low visibility | gate on visibility / a temporal model, or a 3D pose prior |
| Multiple people in view | tracker may switch to another person | `num_poses=1`, glitch jump guard / track IDs, lock onto the calibrated player by size and position |
| Player too close (feet cut off) | no foot targets; hand targets clamp to the frame | foot mode disabled automatically; "step back" prompt during calibration |
| Side-on player | shoulder width shrinks, so scale estimate drops | clamp 0.6–1.6 / use torso length or 3D world landmarks for scale |
| Left and right swap | doesn't matter: effectors are matched by kind | — |

## 9. Taking it to production

- **Runtime.** C++ with the MediaPipe C++ graph, or a TFLite/ONNX model directly, using a GPU/NPU delegate. Zero-copy buffers from camera to inference to render, and a fixed-rate render loop separate from inference.
- **Model.** Distil or fine-tune for the console's camera, viewing distance and target demographics (lots of kids). Quantize to int8, and measure accuracy *per joint at speed*, not just mean error on still poses.
- **Temporal modelling.** Predict ahead by about one inference latency (Kalman or a learned predictor), so hits register on time, not ~50 ms late. Add a short temporal smoothing network instead of hand-tuned filters.
- **Multi-person.** Detection with persistent IDs, and per-player calibration and hit ownership.
- **Telemetry.** Ship the p50/p95 stage timings and dropped-frame counts from the field. Replay files (landmarks or video, with consent) are the main tool for reproducing bugs.
