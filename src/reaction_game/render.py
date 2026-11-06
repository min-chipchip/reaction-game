"""OpenCV drawing: camera feed, skeleton, hotspots, HUD, state screens, debug overlay."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .game import Game, HitEvent, MissEvent, State
from .metrics import Metrics
from .pose import SKELETON_EDGES, Body, Effector
from .session import Session

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (80, 220, 100)
RED = (60, 60, 230)
YELLOW = (40, 210, 250)
KIND_COLOR = {"hand": (255, 200, 40), "foot": (40, 150, 255)}  # BGR: cyan-ish, orange


def text(img, s: str, org: tuple[int, int], scale: float = 0.8, color=WHITE, thick: int = 2,
         center: bool = False) -> None:
    if center:
        (tw, th), _ = cv2.getTextSize(s, FONT, scale, thick)
        org = (int(org[0] - tw / 2), int(org[1] + th / 2))
    cv2.putText(img, s, org, FONT, scale, BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


@dataclass
class _Popup:
    s: str
    x: float
    y: float
    t0: float
    color: tuple[int, int, int]


class Renderer:
    POPUP_SECONDS = 0.7

    def __init__(self):
        self._popups: list[_Popup] = []

    def on_events(self, t: float, events: list) -> None:
        for e in events:
            if isinstance(e, HitEvent):
                self._popups.append(_Popup(f"+{e.record.points}", e.x, e.y, t, GREEN))
            elif isinstance(e, MissEvent):
                self._popups.append(_Popup("miss", e.hotspot.x, e.hotspot.y, t, RED))

    def draw(self, img: np.ndarray, session: Session, t: float, metrics: Metrics | None = None,
             debug: bool = False, extra_debug: list[str] | None = None) -> np.ndarray:
        g = session.game
        if session.body is not None:
            self._skeleton(img, session.body, g.cfg.min_visibility)
        self._hotspots(img, g, t)
        self._effectors(img, session.effectors, g.cfg.min_visibility)
        self._draw_popups(img, t)
        {
            State.CALIBRATING: self._calibrating,
            State.COUNTDOWN: self._countdown,
            State.PLAYING: self._hud,
            State.PAUSED: self._paused,
            State.GAME_OVER: self._game_over,
        }[g.state](img, g, t)
        if debug:
            self._debug(img, session, metrics, extra_debug or [])
        return img

    # ---- layers -----------------------------------------------------------

    def _skeleton(self, img, body: Body, min_vis: float) -> None:
        for i, j in SKELETON_EDGES:
            if body.vis(i) >= min_vis and body.vis(j) >= min_vis:
                a = tuple(int(v) for v in body.xy(i))
                b = tuple(int(v) for v in body.xy(j))
                cv2.line(img, a, b, (230, 230, 230), 3, cv2.LINE_AA)

    def _effectors(self, img, effectors: dict[str, Effector], min_vis: float) -> None:
        for name, e in effectors.items():
            if e.visibility >= min_vis:
                color = KIND_COLOR["hand" if name.startswith("hand") else "foot"]
                cv2.circle(img, (int(e.x), int(e.y)), 12, color, -1, cv2.LINE_AA)
                cv2.circle(img, (int(e.x), int(e.y)), 12, WHITE, 2, cv2.LINE_AA)

    def _hotspots(self, img, g: Game, t: float) -> None:
        if not g.active:
            return
        overlay = img.copy()
        for h in g.active:
            cv2.circle(overlay, (int(h.x), int(h.y)), int(h.radius), KIND_COLOR[h.kind], -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)
        for h in g.active:
            c, r = (int(h.x), int(h.y)), int(h.radius)
            rem = h.remaining(t)
            ring = GREEN if rem > 0.5 else YELLOW if rem > 0.25 else RED
            cv2.circle(img, c, r, WHITE, 2, cv2.LINE_AA)
            # time-left arc, sweeping clockwise from 12 o'clock
            cv2.ellipse(img, c, (r + 8, r + 8), -90, 0, 360 * rem, ring, 6, cv2.LINE_AA)
            text(img, h.kind.upper(), c, 0.55, WHITE, 2, center=True)

    def _draw_popups(self, img, t: float) -> None:
        alive = []
        for p in self._popups:
            age = t - p.t0
            if 0 <= age < self.POPUP_SECONDS:
                text(img, p.s, (int(p.x), int(p.y - 60 * age / self.POPUP_SECONDS)), 1.0, p.color, 2, center=True)
                alive.append(p)
        self._popups = alive

    # ---- state screens ----------------------------------------------------

    def _banner(self, img, lines: list[tuple[str, float]], y_frac: float = 0.18) -> None:
        h, w = img.shape[:2]
        y = int(h * y_frac)
        for s, scale in lines:
            text(img, s, (w // 2, y), scale, WHITE, 2, center=True)
            y += int(48 * scale + 12)

    def _calibrating(self, img, g: Game, t: float) -> None:
        h, w = img.shape[:2]
        self._banner(img, [("Step back so your whole body is visible", 0.9),
                           ("Hold a T-pose to calibrate", 1.1),
                           ("(press S to skip)", 0.6)])
        p = g.calibrator.progress(t)
        x0, x1, y = int(w * 0.3), int(w * 0.7), int(h * 0.88)
        cv2.rectangle(img, (x0, y), (x1, y + 18), WHITE, 2)
        cv2.rectangle(img, (x0, y), (x0 + int((x1 - x0) * p), y + 18), GREEN, -1)

    def _countdown(self, img, g: Game, t: float) -> None:
        h, w = img.shape[:2]
        n = math.ceil(g.countdown_left(t))
        text(img, str(max(n, 1)), (w // 2, h // 2), 5.0, YELLOW, 10, center=True)
        mode = "hands + feet" if g.feet_enabled else "hands"
        self._banner(img, [(f"Get ready - touch the targets ({mode})", 0.9)])

    def _hud(self, img, g: Game, t: float) -> None:
        h, w = img.shape[:2]
        text(img, f"Score {g.score}", (20, 45), 1.1)
        if g.combo >= 2:
            text(img, f"Combo x{g.combo_multiplier:.1f}", (20, 90), 0.8, YELLOW)
        left = g.time_left(t)
        text(img, f"{left:4.1f}s", (w - 150, 45), 1.1, RED if left < 10 else WHITE)

    def _paused(self, img, g: Game, t: float) -> None:
        self._hud(img, g, t)
        self._banner(img, [("PAUSED", 1.6), ("Step back into view", 1.0)], 0.4)

    def _game_over(self, img, g: Game, t: float) -> None:
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (int(w * 0.25), int(h * 0.12)), (int(w * 0.75), int(h * 0.88)), BLACK, -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, dst=img)
        s = g.summary()

        def ms(v):
            return "-" if v is None else f"{v:.0f} ms"

        lines = [
            ("GAME OVER", 1.5),
            (f"Score  {s['score']}", 1.2),
            (f"Hits {s['hits']}   Misses {s['misses']}   Accuracy {s['accuracy'] * 100:.0f}%", 0.75),
            (f"Reaction  avg {ms(s['mean_reaction_ms'])}   best {ms(s['best_reaction_ms'])}", 0.75),
            (f"Best combo  {s['best_combo']}", 0.75),
            ("R: play again    Q: quit", 0.7),
        ]
        self._banner(img, lines, 0.22)

    # ---- debug --------------------------------------------------------------

    def _debug(self, img, session: Session, metrics: Metrics | None, extra: list[str]) -> None:
        h, w = img.shape[:2]
        min_vis = session.game.cfg.min_visibility
        # raw (red) vs smoothed (green) effectors: shows what the One Euro filter is doing
        for name, e in session.raw_effectors.items():
            if e.visibility >= min_vis:
                cv2.circle(img, (int(e.x), int(e.y)), 4, RED, -1)
                s = session.effectors.get(name)
                if s is not None:
                    cv2.circle(img, (int(s.x), int(s.y)), 4, GREEN, -1)
        lines = list(extra)
        if metrics is not None:
            lines.insert(0, f"FPS {metrics.fps:5.1f}")
            lines += [f"{n:<10} p50 {p50:5.1f}  p95 {p95:5.1f} ms" for n, p50, p95 in metrics.report()]
        if session.body is not None:
            vis = " ".join(f"{n}:{e.visibility:.2f}" for n, e in session.raw_effectors.items())
            lines.append(vis)
        panel_h = 26 * len(lines) + 16
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (min(w, 620), h), BLACK, -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)
        for i, s in enumerate(lines):
            cv2.putText(img, s, (10, h - panel_h + 26 + 26 * i), cv2.FONT_HERSHEY_PLAIN, 1.3, WHITE, 1, cv2.LINE_AA)
