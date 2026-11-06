"""Pose tracking: MediaPipe PoseLandmarker -> Body (pixel-space landmarks).

The rest of the game never sees MediaPipe types. It only sees `Body`, which can
come from the live tracker, a recording, or a synthetic test fixture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

# MediaPipe Pose landmark indices (33-point BlazePose topology).
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_INDEX, R_INDEX = 19, 20
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32
NUM_LANDMARKS = 33

# Each effector is the midpoint of two landmarks. Its visibility is the lower of
# the two visibilities.
#   hand = between the wrist and the index fingertip (roughly the palm).
#        The fingertip alone is noisy; the wrist alone undershoots reach.
#   foot = between the ankle and the toe.
EFFECTOR_LANDMARKS: dict[str, tuple[int, int]] = {
    "hand_l": (L_WRIST, L_INDEX),
    "hand_r": (R_WRIST, R_INDEX),
    "foot_l": (L_ANKLE, L_FOOT),
    "foot_r": (R_ANKLE, R_FOOT),
}

# Body-only skeleton edges for drawing (face omitted).
SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (15, 19), (16, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)


def effector_kind(name: str) -> str:
    """'hand_l' -> 'hand'."""
    return name.split("_", 1)[0]


@dataclass(frozen=True)
class Effector:
    x: float
    y: float
    visibility: float


@dataclass
class Body:
    """One tracked person: 33 landmarks as (x_px, y_px, visibility), in the mirrored display frame."""

    landmarks: list[tuple[float, float, float]]

    def xy(self, i: int) -> tuple[float, float]:
        x, y, _ = self.landmarks[i]
        return x, y

    def vis(self, i: int) -> float:
        return self.landmarks[i][2]

    def mid(self, i: int, j: int) -> tuple[float, float]:
        (ax, ay), (bx, by) = self.xy(i), self.xy(j)
        return (ax + bx) / 2.0, (ay + by) / 2.0

    def shoulder_center(self) -> tuple[float, float]:
        return self.mid(L_SHOULDER, R_SHOULDER)

    def shoulder_width(self) -> float:
        (ax, ay), (bx, by) = self.xy(L_SHOULDER), self.xy(R_SHOULDER)
        return math.hypot(ax - bx, ay - by)

    def shoulders_visible(self, min_vis: float) -> bool:
        return self.vis(L_SHOULDER) >= min_vis and self.vis(R_SHOULDER) >= min_vis

    def effectors(self) -> dict[str, Effector]:
        out: dict[str, Effector] = {}
        for name, (i, j) in EFFECTOR_LANDMARKS.items():
            x, y = self.mid(i, j)
            out[name] = Effector(x, y, min(self.vis(i), self.vis(j)))
        return out

    def to_json(self) -> list[list[float]]:
        # Full precision on purpose. The live game ran on these exact floats, so
        # rounding here would make a replay drift from the recorded game.
        return [[x, y, v] for x, y, v in self.landmarks]

    @classmethod
    def from_json(cls, data: list[list[float]]) -> "Body":
        return cls([(float(x), float(y), float(v)) for x, y, v in data])


MODEL_FILES = {
    "lite": "pose_landmarker_lite.task",
    "full": "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
}
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
)


def model_path(variant: str, models_dir: Path) -> Path:
    return models_dir / MODEL_FILES[variant]


class PoseTracker:
    """Thin wrapper around MediaPipe's PoseLandmarker in VIDEO mode.

    VIDEO mode (not IMAGE) lets MediaPipe track between frames. After the first
    detection it skips the person detector and follows the previous pose, which
    is much cheaper. The trade-off is that timestamps must strictly increase.
    """

    def __init__(self, model_file: Path, min_detection: float = 0.5, min_tracking: float = 0.5):
        import mediapipe as mp  # imported lazily so tests and replays don't need it
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_file)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection,
            min_pose_presence_confidence=min_detection,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, frame_bgr, ts: float) -> Body | None:
        """Run pose on a BGR frame captured at monotonic time `ts` (seconds)."""
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = max(int(ts * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.pose_landmarks:
            return None
        lms = result.pose_landmarks[0]
        return Body([(lm.x * w, lm.y * h, float(lm.visibility or 0.0)) for lm in lms])

    def close(self) -> None:
        self._landmarker.close()
