"""Threaded webcam capture that always hands back the newest frame.

`cv2.VideoCapture.read()` returns frames from a driver queue. If the consumer
(pose inference + render) is slower than the camera, reading in the main loop
falls further and further behind, and the player sees their movements late. A
reader thread that keeps only the newest frame bounds that lag to about one
frame. Frames the consumer never sees are counted in `dropped`.
"""
from __future__ import annotations

import threading
import time

import cv2


class CameraError(RuntimeError):
    pass


class CameraStream:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.index = index
        self._req = (width, height, fps)
        self._cap: cv2.VideoCapture | None = None
        self._cond = threading.Condition()
        self._frame = None
        self._ts = 0.0
        self._seq = 0
        self._last_read_seq = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self.dropped = 0
        self.frame_size: tuple[int, int] = (width, height)

    def start(self) -> "CameraStream":
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise CameraError(
                f"Could not open camera {self.index}. On macOS, allow camera access for your "
                "terminal app in System Settings > Privacy & Security > Camera, then restart it."
            )
        w, h, fps = self._req
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best effort; not all backends honour it
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise CameraError(
                f"Camera {self.index} opened but returned no frames (permission denied or in use?)."
            )
        self.frame_size = (frame.shape[1], frame.shape[0])
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        assert self._cap is not None
        while self._running:
            ok, frame = self._cap.read()
            ts = time.monotonic()  # approximate capture time; excludes sensor exposure/USB transfer
            if not ok:
                time.sleep(0.005)
                continue
            with self._cond:
                self._frame, self._ts = frame, ts
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 1.0):
        """Block until a frame newer than the last one returned arrives.

        Returns (frame, capture_ts, seq), or None on timeout.
        """
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > self._last_read_seq, timeout):
                return None
            skipped = self._seq - self._last_read_seq - 1
            if self._last_read_seq and skipped > 0:
                self.dropped += skipped
            self._last_read_seq = self._seq
            return self._frame, self._ts, self._seq

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
