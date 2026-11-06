"""Record a session as landmark JSONL and replay it without a camera.

File format, one JSON object per line:
    {"type": "header", "version": 1, "frame_size": [w, h], "config": {...}}
    {"type": "frame", "t": 12.345, "body": [[x, y, vis] * 33] | null}
    {"type": "key", "t": 12.345, "key": "s"}

Replaying through `Session` reproduces the original score exactly. This is
useful for regression tests, for debugging a reported bug, and for tuning
filter or hit-test parameters on real movement without having to play again.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .pose import Body
from .session import Session, SessionConfig


class Recorder:
    def __init__(self, path: Path, config: SessionConfig, frame_size: tuple[int, int]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("w")
        self._write({"type": "header", "version": 1, "frame_size": list(frame_size),
                     "config": config.to_dict()})

    def _write(self, obj: dict) -> None:
        self._f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def frame(self, t: float, body: Body | None) -> None:
        self._write({"type": "frame", "t": t, "body": body.to_json() if body else None})

    def key(self, t: float, key: str) -> None:
        self._write({"type": "key", "t": t, "key": key})

    def close(self) -> None:
        self._f.close()


def load(path: Path) -> tuple[dict, Iterator[dict]]:
    f = path.open()
    header = json.loads(f.readline())
    if header.get("type") != "header":
        raise ValueError(f"{path}: first line is not a header")

    def records() -> Iterator[dict]:
        with f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    return header, records()


def session_from_header(header: dict) -> Session:
    return Session(SessionConfig.from_dict(header["config"]), tuple(header["frame_size"]))


def apply(session: Session, rec: dict) -> list:
    """Feed one recorded line into the session. Returns that frame's events."""
    if rec["type"] == "frame":
        body = Body.from_json(rec["body"]) if rec["body"] is not None else None
        return session.step(rec["t"], body)
    if rec["type"] == "key":
        session.key(rec["t"], rec["key"])
    return []


def replay_headless(path: Path) -> Session:
    header, records = load(path)
    session = session_from_header(header)
    for rec in records:
        apply(session, rec)
    return session
