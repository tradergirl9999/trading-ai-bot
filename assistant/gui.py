"""
Jarvis GUI v2 — Sci-fi neural mesh face
Run: python gui.py
"""
import os, sys, math, random, time, threading, queue, datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
if not os.getenv("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY not set — add it to assistant/.env")

import tkinter as tk

from brain import JarvisAssistant
from voice import VoiceIO
from tools import reminders

# ── Palette ──────────────────────────────────────────────────────────────────
BG          = "#030c0e"
TEAL_BRT    = "#00e5c8"
TEAL_MID    = "#00aa88"
TEAL_DIM    = "#004433"
TEAL_LINE   = "#005544"
CYAN        = "#00bfff"
CYAN_DIM    = "#003366"
ORANGE      = "#ff8800"
ORANGE_DIM  = "#331100"
WHITE       = "#c8e0dc"
TEXT_DIM    = "#304848"
CHAT_BG     = "#040f10"
CHAT_BORDER = "#006655"
STATUS_BG   = "#040e10"

FONT_MONO  = ("Consolas", 10)
FONT_SMALL = ("Consolas", 9)
FONT_TITLE = ("Consolas", 12, "bold")
FONT_CHAT  = ("Consolas", 10)

# ── Face node definitions: (bx, by, radius, color, is_special) ──────────────
# bx, by are fractions of the FACE CANVAS region (not full window)
_FACE_NODES_DEF = [
    # Head oval
    (0.50, 0.10, 2.0, TEAL_MID,  False),
    (0.38, 0.17, 1.8, TEAL_MID,  False),
    (0.29, 0.29, 1.8, TEAL_MID,  False),
    (0.26, 0.43, 1.8, TEAL_MID,  False),
    (0.30, 0.57, 1.8, TEAL_MID,  False),
    (0.38, 0.68, 1.8, TEAL_MID,  False),
    (0.50, 0.72, 2.0, TEAL_MID,  False),
    (0.62, 0.68, 1.8, TEAL_MID,  False),
    (0.70, 0.57, 1.8, TEAL_MID,  False),
    (0.74, 0.43, 1.8, TEAL_MID,  False),
    (0.71, 0.29, 1.8, TEAL_MID,  False),
    (0.62, 0.17, 1.8, TEAL_MID,  False),
    # Forehead structure
    (0.50, 0.21, 2.2, TEAL_BRT,  False),
    (0.42, 0.24, 1.5, TEAL_MID,  False),
    (0.58, 0.24, 1.5, TEAL_MID,  False),
    # Left eye (special — large glow)
    (0.38, 0.36, 5.5, CYAN,      True),
    (0.33, 0.38, 1.5, TEAL_MID,  False),
    (0.44, 0.33, 1.5, TEAL_MID,  False),
    (0.38, 0.41, 1.2, TEAL_LINE, False),
    # Right eye (special — large glow)
    (0.62, 0.36, 5.5, CYAN,      True),
    (0.67, 0.38, 1.5, TEAL_MID,  False),
    (0.56, 0.33, 1.5, TEAL_MID,  False),
    (0.62, 0.41, 1.2, TEAL_LINE, False),
    # Nose bridge & tip
    (0.50, 0.40, 2.0, TEAL_BRT,  False),
    (0.46, 0.47, 1.5, TEAL_MID,  False),
    (0.54, 0.47, 1.5, TEAL_MID,  False),
    (0.50, 0.51, 2.5, TEAL_MID,  False),
    # Mouth
    (0.43, 0.60, 1.5, TEAL_MID,  False),
    (0.47, 0.62, 2.0, TEAL_BRT,  False),
    (0.50, 0.63, 2.5, TEAL_BRT,  True),
    (0.53, 0.62, 2.0, TEAL_BRT,  False),
    (0.57, 0.60, 1.5, TEAL_MID,  False),
    # Cheekbones
    (0.34, 0.50, 2.0, TEAL_MID,  False),
    (0.66, 0.50, 2.0, TEAL_MID,  False),
    # Interior structure
    (0.44, 0.44, 1.5, TEAL_DIM,  False),
    (0.56, 0.44, 1.5, TEAL_DIM,  False),
    (0.50, 0.30, 1.8, TEAL_MID,  False),
    (0.42, 0.54, 1.5, TEAL_DIM,  False),
    (0.58, 0.54, 1.5, TEAL_DIM,  False),
]

# Scattered nebula nodes (far-field connections, relative to full canvas)
_NEBULA_DEF = [
    (0.06, 0.12, ORANGE),  (0.12, 0.06, TEAL_MID),
    (0.04, 0.42, TEAL_MID),(0.08, 0.70, ORANGE),
    (0.14, 0.88, TEAL_MID),(0.06, 0.92, ORANGE_DIM),
    (0.94, 0.12, TEAL_MID),(0.88, 0.06, ORANGE),
    (0.96, 0.42, TEAL_MID),(0.92, 0.70, ORANGE),
    (0.86, 0.88, TEAL_MID),(0.94, 0.92, ORANGE_DIM),
    (0.50, 0.97, TEAL_MID),(0.30, 0.94, TEAL_DIM),
    (0.70, 0.94, TEAL_DIM),(0.18, 0.04, TEAL_DIM),
    (0.50, 0.02, TEAL_MID),(0.82, 0.04, TEAL_DIM),
    (0.20, 0.50, TEAL_DIM),(0.80, 0.50, TEAL_DIM),
]

CONN_THRESHOLD  = 0.22   # fraction of canvas — max distance to draw an edge
FACE_AREA       = (0.25, 0.08, 0.75, 0.92)  # (left, top, right, bottom) fractions of canvas


def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def _tri_fill_color(base: str) -> str:
    r = int(base[1:3], 16)
    g = int(base[3:5], 16)
    b = int(base[5:7], 16)
    # very faint — 5% opacity approximation against #030c0e bg
    br, bg_c, bb = 3, 12, 14
    a = 0.055
    nr = int(br + (r - br) * a)
    ng = int(bg_c + (g - bg_c) * a)
    nb = int(bb + (b - bb) * a)
    return f"#{nr:02x}{ng:02x}{nb:02x}"


class NeuralFaceCanvas(tk.Canvas):
    """Animated polygonal mesh face."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, highlightthickness=0, **kw)
        self._t = 0.0
        self._state = "idle"
        self._running = True
        self._nodes: list[dict] = []     # face nodes (canvas-space)
        self._nebula: list[dict] = []    # outer nodes (canvas-space)
        self._triangles: list[tuple] = []
        self._pulses: list[dict] = []    # data-flow animations
        self._chat_lines: list[dict] = []  # {sender, text, ts}
        self._w = 0
        self._h = 0
        self.bind("<Configure>", self._on_resize)

    # ── Layout helpers ────────────────────────────────────────────────────────

    def _face_rect(self):
        l, t, r, b = FACE_AREA
        return int(self._w * l), int(self._h * t), int(self._w * r), int(self._h * b)

    def _face_coord(self, bx, by):
        fl, ft, fr, fb = self._face_rect()
        fw = fr - fl
        fh = fb - ft
        return fl + bx * fw, ft + by * fh

    def _canvas_coord(self, bx, by):
        return bx * self._w, by * self._h

    def _on_resize(self, event):
        self._w, self._h = event.width, event.height
        self._build_nodes()

    def _build_nodes(self):
        if self._w < 10 or self._h < 10:
            return
        self._nodes = []
        for bx, by, r, color, special in _FACE_NODES_DEF:
            x, y = self._face_coord(bx, by)
            self._nodes.append({
                "bx": bx, "by": by, "x": x, "y": y,
                "r": r, "color": color, "special": special,
                "phase": random.uniform(0, math.pi * 2),
                "speed": random.uniform(0.008, 0.018),
            })

        self._nebula = []
        for bx, by, color in _NEBULA_DEF:
            x, y = self._canvas_coord(bx, by)
            self._nebula.append({
                "bx": bx, "by": by, "x": x, "y": y,
                "r": random.uniform(2.5, 4.5),
                "color": color,
                "phase": random.uniform(0, math.pi * 2),
                "speed": random.uniform(0.004, 0.012),
                "halo": random.random() < 0.5,
            })

        # Precompute triangles from face nodes
        all_pts = [(n["bx"], n["by"]) for n in self._nodes]
        # Use face-area fractional coords scaled to [0,1] for threshold
        self._triangles = []
        n = len(all_pts)
        thr = 0.20
        for i in range(n):
            for j in range(i + 1, n):
                if _dist(*all_pts[i], *all_pts[j]) < thr:
                    for k in range(j + 1, n):
                        if (_dist(*all_pts[i], *all_pts[k]) < thr and
                                _dist(*all_pts[j], *all_pts[k]) < thr):
                            self._triangles.append((i, j, k))

        # Seed some pulses
        self._pulses = []
        for _ in range(6):
            self._spawn_pulse()

    def _spawn_pulse(self):
        if len(self._nodes) < 2:
            return
        src = random.randint(0, len(self._nodes) - 1)
        tgt = random.randint(0, len(self._nodes) - 1)
        while tgt == src:
            tgt = random.randint(0, len(self._nodes) - 1)
        self._pulses.append({
            "src": src, "tgt": tgt, "progress": 0.0,
            "speed": random.uniform(0.015, 0.04),
            "color": random.choice([TEAL_BRT, CYAN, ORANGE]),
        })

    # ── State ─────────────────────────────────────────────────────────────────

    def set_state(self, state: str):
        self._state = state

    def add_chat_line(self, sender: str, text: str):
        ts = datetime.datetime.now().strftime("%H:%M")
        # Wrap long text
        max_chars = 38
        words = text.split()
        lines, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 <= max_chars:
                cur = (cur + " " + w).strip()
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        self._chat_lines.append({"sender": sender, "lines": lines, "ts": ts})
        if len(self._chat_lines) > 18:
            self._chat_lines = self._chat_lines[-18:]

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _animated_pos(self, node: dict, t: float):
        fl, ft, fr, fb = self._face_rect()
        fw = fr - fl
        fh = fb - ft
        dx = math.sin(t * node["speed"] + node["phase"]) * fw * 0.018
        dy = math.cos(t * node["speed"] * 0.7 + node["phase"]) * fh * 0.014
        return node["x"] + dx, node["y"] + dy

    def _nebula_pos(self, node: dict, t: float):
        dx = math.sin(t * node["speed"] + node["phase"]) * self._w * 0.015
        dy = math.cos(t * node["speed"] * 0.8 + node["phase"]) * self._h * 0.012
        return node["x"] + dx, node["y"] + dy

    def _draw_frame(self):
        if self._w < 10 or not self._nodes:
            return
        self.delete("all")
        t = self._t

        # 1. Background radial glow in face center
        cx = self._w * 0.5
        cy = self._h * 0.45
        for i, rp in enumerate([0.55, 0.45, 0.35, 0.25, 0.15]):
            r = min(self._w, self._h) * rp
            alpha = [10, 8, 6, 4, 3][i]
            c = f"#{0:02x}{alpha:02x}{alpha//2:02x}"
            self.create_oval(cx-r, cy-r, cx+r, cy+r, fill=c, outline="")

        # 2. Circuit lines (corners)
        self._draw_circuit_lines(t)

        # 3. Nebula nodes + their connections
        neb_pos = [self._nebula_pos(n, t) for n in self._nebula]
        face_pos = [self._animated_pos(n, t) for n in self._nodes]

        for i, (nx, ny) in enumerate(neb_pos):
            # Connect nebula to nearest face nodes
            for j, (fx, fy) in enumerate(face_pos):
                d = _dist(nx / self._w, ny / self._h, fx / self._w, fy / self._h)
                if d < 0.18:
                    pulse_alpha = int(30 + 15 * math.sin(t * 0.5 + i))
                    lc = f"#{0:02x}{pulse_alpha:02x}{pulse_alpha//2:02x}"
                    self.create_line(nx, ny, fx, fy, fill=lc, width=1)
            # Connect nebula nodes to each other if nearby
            for k in range(i + 1, len(neb_pos)):
                kx, ky = neb_pos[k]
                d = _dist(nx / self._w, ny / self._h, kx / self._w, ky / self._h)
                if d < 0.14:
                    lc = TEAL_DIM
                    self.create_line(nx, ny, kx, ky, fill=lc, width=1, dash=(3, 6))

        # 4. Face triangles (faint fill)
        for (i, j, k) in self._triangles:
            ix, iy = face_pos[i]
            jx, jy = face_pos[j]
            kx, ky = face_pos[k]
            # Vary brightness slightly per triangle
            pulse = 0.5 + 0.5 * math.sin(t * 0.3 + i * 0.4)
            color_idx = [TEAL_MID, TEAL_LINE, CYAN][i % 3]
            fill = _tri_fill_color(color_idx)
            self.create_polygon(ix, iy, jx, jy, kx, ky,
                                fill=fill, outline="")

        # 5. Face edges
        face_thr_px = min(self._w, self._h) * CONN_THRESHOLD
        for i, (ix, iy) in enumerate(face_pos):
            for j in range(i + 1, len(face_pos)):
                jx, jy = face_pos[j]
                d = math.hypot(ix - jx, iy - jy)
                if d < face_thr_px:
                    brightness = int(40 + 30 * math.sin(t * 0.4 + i * 0.3))
                    lc = f"#{0:02x}{brightness:02x}{brightness//2:02x}"
                    self.create_line(ix, iy, jx, jy, fill=lc, width=1)

        # 6. Data pulses along face edges
        for pulse in self._pulses:
            if pulse["src"] >= len(face_pos) or pulse["tgt"] >= len(face_pos):
                continue
            sx, sy = face_pos[pulse["src"]]
            ex, ey = face_pos[pulse["tgt"]]
            p = pulse["progress"]
            px = sx + (ex - sx) * p
            py = sy + (ey - sy) * p
            r = 3.5
            self.create_oval(px - r, py - r, px + r, py + r,
                             fill=pulse["color"], outline="")
            # Glow
            r2 = 7
            self.create_oval(px - r2, py - r2, px + r2, py + r2,
                             fill="", outline=pulse["color"])

        # 7. Nebula node dots + halos
        for i, (nx, ny) in enumerate(neb_pos):
            nd = self._nebula[i]
            r = nd["r"]
            pulse = 0.6 + 0.4 * math.sin(t * nd["speed"] * 40 + nd["phase"])
            if nd["halo"]:
                hr = r * 3.5 * pulse
                self.create_oval(nx-hr, ny-hr, nx+hr, ny+hr,
                                 fill="", outline=nd["color"], width=1)
            self.create_oval(nx-r, ny-r, nx+r, ny+r,
                             fill=nd["color"], outline="")

        # 8. Face nodes + halos
        for i, (fx, fy) in enumerate(face_pos):
            nd = self._nodes[i]
            r = nd["r"]
            color = nd["color"]
            pulse = 0.7 + 0.3 * math.sin(t * nd["speed"] * 30 + nd["phase"])

            if nd["special"]:
                # Eye / key nodes: larger halo rings
                for hr_mult, alpha_base in [(6, 40), (4, 70), (2.5, 100)]:
                    hr = r * hr_mult * pulse
                    c = color if alpha_base > 60 else TEAL_DIM
                    self.create_oval(fx-hr, fy-hr, fx+hr, fy+hr,
                                     fill="", outline=c, width=1)
                # Inner fill with glow
                self.create_oval(fx-r*1.5, fy-r*1.5, fx+r*1.5, fy+r*1.5,
                                 fill=color, outline="")
            else:
                if r > 1.8:
                    hr = r * 2.5 * pulse
                    self.create_oval(fx-hr, fy-hr, fx+hr, fy+hr,
                                     fill="", outline=TEAL_LINE, width=1)
                self.create_oval(fx-r, fy-r, fx+r, fy+r,
                                 fill=color, outline="")

        # 9. State-driven overlay
        self._draw_state_overlay(t)

        # 10. Chat panel
        self._draw_chat_panel()

        # 11. Top status bar
        self._draw_status_bar()

    def _draw_circuit_lines(self, t: float):
        w, h = self._w, self._h
        margin = 30
        pulse = int(20 + 15 * math.sin(t * 0.3))
        lc = f"#{0:02x}{pulse:02x}{pulse//2:02x}"

        # Four corner circuit segments
        segs = [
            # top-left
            [(margin, margin), (margin + 60, margin), (margin + 60, margin + 30), (margin + 90, margin + 30)],
            [(margin, margin), (margin, margin + 60), (margin + 30, margin + 60), (margin + 30, margin + 90)],
            # top-right
            [(w-margin, margin), (w-margin-60, margin), (w-margin-60, margin+30), (w-margin-90, margin+30)],
            [(w-margin, margin), (w-margin, margin+60), (w-margin-30, margin+60), (w-margin-30, margin+90)],
            # bottom-left
            [(margin, h-margin), (margin+60, h-margin), (margin+60, h-margin-30), (margin+90, h-margin-30)],
            [(margin, h-margin), (margin, h-margin-60), (margin+30, h-margin-60), (margin+30, h-margin-90)],
            # bottom-right
            [(w-margin, h-margin), (w-margin-60, h-margin), (w-margin-60, h-margin-30), (w-margin-90, h-margin-30)],
            [(w-margin, h-margin), (w-margin, h-margin-60), (w-margin-30, h-margin-60), (w-margin-30, h-margin-90)],
        ]
        for seg in segs:
            flat = [v for pt in seg for v in pt]
            self.create_line(*flat, fill=lc, width=1)
            # dot at end
            ex, ey = seg[-1]
            r = 2
            self.create_oval(ex-r, ey-r, ex+r, ey+r, fill=TEAL_MID, outline="")

    def _draw_state_overlay(self, t: float):
        cx = self._w * 0.5
        cy = self._h * 0.45
        state = self._state

        if state == "listening":
            # Concentric expanding rings
            for i in range(3):
                phase = t * 1.5 + i * (math.pi * 2 / 3)
                r = 80 + 30 * math.sin(phase)
                alpha = int(80 + 60 * math.sin(phase))
                ac = f"#{0:02x}{min(255,alpha):02x}{min(255,alpha//2):02x}"
                self.create_oval(cx-r, cy-r, cx+r, cy+r,
                                 fill="", outline=ac, width=2)
            # Label
            self.create_text(cx, self._h - 55, text="● LISTENING",
                             fill=TEAL_BRT, font=FONT_MONO)

        elif state == "thinking":
            # Rotating arc
            segments = 12
            for i in range(segments):
                angle = t * 3 + i * (2 * math.pi / segments)
                r1, r2 = 70, 85
                x1 = cx + r1 * math.cos(angle)
                y1 = cy + r1 * math.sin(angle)
                x2 = cx + r2 * math.cos(angle)
                y2 = cy + r2 * math.sin(angle)
                brightness = int(100 + 155 * ((i / segments + t * 0.3) % 1.0))
                lc = f"#{0:02x}{min(255,brightness//2):02x}{min(255,brightness):02x}"
                self.create_line(x1, y1, x2, y2, fill=lc, width=2)
            self.create_text(cx, self._h - 55, text="◌ PROCESSING",
                             fill=CYAN, font=FONT_MONO)

        elif state == "speaking":
            # Audio bars
            bars = 9
            bar_w = 8
            spacing = 5
            total = bars * bar_w + (bars - 1) * spacing
            x0 = cx - total // 2
            for i in range(bars):
                h_bar = 12 + 28 * abs(math.sin(t * 5 + i * 0.9))
                x = x0 + i * (bar_w + spacing)
                alpha = int(180 + 75 * math.sin(t * 4 + i))
                bc = f"#{0:02x}{min(255,alpha):02x}{min(255,alpha//2):02x}"
                self.create_rectangle(x, cy - h_bar, x + bar_w, cy + h_bar,
                                      fill=bc, outline="")
            self.create_text(cx, self._h - 55, text="◈ SPEAKING",
                             fill=TEAL_BRT, font=FONT_MONO)

        elif state == "error":
            r = 60 + 10 * math.sin(t * 4)
            self.create_oval(cx-r, cy-r, cx+r, cy+r,
                             fill="", outline="#ff4444", width=2)
            self.create_text(cx, self._h - 55, text="⚠ ERROR",
                             fill="#ff4444", font=FONT_MONO)

        else:
            # Idle — subtle breathing ring
            r = 60 + 5 * math.sin(t * 0.8)
            alpha = int(25 + 15 * math.sin(t * 0.8))
            ac = f"#{0:02x}{alpha:02x}{alpha//2:02x}"
            self.create_oval(cx-r, cy-r, cx+r, cy+r,
                             fill="", outline=ac, width=1)
            self.create_text(cx, self._h - 55, text="○ STANDBY",
                             fill=TEXT_DIM, font=FONT_MONO)

    def _draw_chat_panel(self):
        if not self._chat_lines:
            return
        w, h = self._w, self._h
        # Panel on bottom-right
        panel_w = min(360, int(w * 0.42))
        panel_h = min(320, int(h * 0.52))
        px = w - panel_w - 20
        py = h - panel_h - 70

        # Background
        self.create_rectangle(px, py, px + panel_w, py + panel_h,
                              fill=CHAT_BG, outline=TEAL_DIM, width=1)
        # Header
        self.create_text(px + 12, py + 12, text="◈ JARVIS COMM",
                         fill=TEAL_BRT, font=("Consolas", 8, "bold"), anchor="w")
        self.create_line(px, py + 24, px + panel_w, py + 24,
                         fill=TEAL_DIM, width=1)

        y_cursor = py + 32
        line_h = 16
        available_h = panel_h - 40

        # Estimate how many lines fit
        max_lines_total = available_h // line_h
        # Flatten all chat content
        flat = []
        for entry in reversed(self._chat_lines):
            prefix = "YOU  " if entry["sender"] == "you" else "J    "
            color  = TEAL_BRT if entry["sender"] == "you" else WHITE
            for ln in reversed(entry["lines"]):
                flat.append((f"{prefix}{ln}", color))
            if flat:
                flat[-1] = (f"{entry['ts']} {flat[-1][0]}", TEXT_DIM)
            if len(flat) >= max_lines_total:
                break

        flat = flat[:max_lines_total]
        flat.reverse()

        for text, color in flat:
            if y_cursor + line_h > py + panel_h - 4:
                break
            self.create_text(px + 10, y_cursor, text=text,
                             fill=color, font=("Consolas", 9), anchor="w")
            y_cursor += line_h

    def _draw_status_bar(self):
        w = self._w
        # Top bar background
        self.create_rectangle(0, 0, w, 38, fill=STATUS_BG, outline="")
        self.create_line(0, 38, w, 38, fill=TEAL_DIM, width=1)
        # Title
        self.create_text(18, 19, text="◈  JARVIS", fill=TEAL_BRT,
                         font=FONT_TITLE, anchor="w")
        # Time
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.create_text(w // 2, 19, text=ts, fill=TEXT_DIM, font=FONT_SMALL)

    # ── Animation loop ────────────────────────────────────────────────────────

    def start(self):
        self._loop()

    def _loop(self):
        if not self._running:
            return
        if self._w > 10 and self._nodes:
            self._draw_frame()

            # Advance pulses
            dead = []
            for p in self._pulses:
                p["progress"] += p["speed"]
                if p["progress"] >= 1.0:
                    dead.append(p)
            for d in dead:
                self._pulses.remove(d)
                self._spawn_pulse()

        self._t += 0.06
        self.after(50, self._loop)

    def stop(self):
        self._running = False


# ── Main App ──────────────────────────────────────────────────────────────────

class JarvisApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("JARVIS")
        self.root.configure(bg=BG)
        self.root.geometry("900x680")
        self.root.minsize(700, 540)

        self.assistant = JarvisAssistant()
        self.voice = VoiceIO()
        self.voice_active = True
        self._q: queue.Queue = queue.Queue()

        reminders.set_speak_callback(self._speak)

        self._build_ui()
        self.canvas.start()
        self.root.after(80, self._process_queue)
        self.root.after(700, self._start_voice_thread)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # Canvas fills the whole window
        self.canvas = NeuralFaceCanvas(self.root)
        self.canvas.pack(fill="both", expand=True)

        # Overlaid controls (placed over canvas)
        # Voice toggle
        self.btn_voice = tk.Button(
            self.root, text="MIC ●", font=("Consolas", 8), bg=STATUS_BG, fg=TEAL_BRT,
            activebackground=BG, activeforeground=TEAL_BRT,
            relief="flat", bd=0, padx=10, pady=4, cursor="hand2",
            command=self._toggle_voice,
        )
        self.btn_voice.place(relx=1.0, rely=0.0, x=-90, y=8, anchor="ne")

        # Reset
        btn_reset = tk.Button(
            self.root, text="RESET", font=("Consolas", 8), bg=STATUS_BG, fg=TEXT_DIM,
            activebackground=BG, activeforeground=TEAL_BRT,
            relief="flat", bd=0, padx=10, pady=4, cursor="hand2",
            command=self._reset,
        )
        btn_reset.place(relx=1.0, rely=0.0, x=-155, y=8, anchor="ne")

        # Input bar
        input_frame = tk.Frame(self.root, bg=BG, bd=0)
        input_frame.place(relx=0, rely=1.0, x=0, y=-48, relwidth=1.0, anchor="sw")

        tk.Frame(input_frame, bg=TEAL_DIM, height=1).pack(fill="x")

        row = tk.Frame(input_frame, bg=BG)
        row.pack(fill="x")

        self.entry = tk.Entry(
            row, bg="#060f10", fg=WHITE, font=("Consolas", 11),
            relief="flat", bd=0, insertbackground=TEAL_BRT,
        )
        self.entry.pack(side="left", fill="x", expand=True, padx=14, pady=10, ipady=5)
        self.entry.bind("<Return>", lambda e: self._on_enter())

        send_btn = tk.Button(
            row, text="SEND ▶", font=("Consolas", 9), bg=TEAL_MID, fg=BG,
            activebackground=TEAL_BRT, activeforeground=BG,
            relief="flat", bd=0, padx=14, pady=8, cursor="hand2",
            command=self._on_enter,
        )
        send_btn.pack(side="right", padx=10, pady=8)

        # Quick-action buttons
        quick = [
            ("Markets", "Give me a quick market overview"),
            ("BTC",     "What is Bitcoin doing right now?"),
            ("News",    "What are today's top financial news stories?"),
            ("Weather", "What's the weather like today?"),
        ]
        qf = tk.Frame(self.root, bg=BG)
        qf.place(relx=0, rely=1.0, x=14, y=-54, anchor="sw")
        for label, prompt in quick:
            b = tk.Button(
                qf, text=label, font=("Consolas", 8), bg="#060f10", fg=TEXT_DIM,
                activebackground=BG, activeforeground=TEAL_BRT,
                relief="flat", bd=0, padx=9, pady=3, cursor="hand2",
                command=lambda p=prompt: threading.Thread(
                    target=self._handle_input, args=(p,), daemon=True).start()
            )
            b.pack(side="left", padx=3)

    # ── Queue ─────────────────────────────────────────────────────────────────

    def _process_queue(self):
        while not self._q.empty():
            cmd, *args = self._q.get_nowait()
            if cmd == "state":
                self.canvas.set_state(args[0])
            elif cmd == "chat":
                self.canvas.add_chat_line(*args)
        self.root.after(60, self._process_queue)

    # ── Voice ──────────────────────────────────────────────────────────────────

    def _start_voice_thread(self):
        threading.Thread(target=self._voice_loop, daemon=True).start()

    def _voice_loop(self):
        while True:
            if not self.voice_active:
                time.sleep(0.4)
                continue
            self._q.put(("state", "listening"))
            text = self.voice.listen()
            if text:
                self._handle_input(text)
            else:
                self._q.put(("state", "idle"))

    def _handle_input(self, text: str):
        cmd = text.lower().strip()
        if cmd in ("quit", "exit", "goodbye", "bye"):
            self._q.put(("chat", "you", text))
            self._q.put(("state", "speaking"))
            self._speak("Goodbye. Stay sharp.")
            self._q.put(("state", "idle"))
            return
        if cmd == "reset":
            self._reset()
            return

        self._q.put(("chat", "you", text))
        self._q.put(("state", "thinking"))

        def _run():
            try:
                response = self.assistant.process(text)
                self._q.put(("chat", "jarvis", response))
                self._q.put(("state", "speaking"))
                self._speak(response)
            except Exception as e:
                self._q.put(("chat", "jarvis", f"Error: {e}"))
                self._q.put(("state", "error"))
                time.sleep(1.5)
            finally:
                self._q.put(("state", "listening" if self.voice_active else "idle"))

        threading.Thread(target=_run, daemon=True).start()

    def _speak(self, text: str):
        self.voice.speak(text)

    def _on_enter(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        threading.Thread(target=self._handle_input, args=(text,), daemon=True).start()

    def _toggle_voice(self):
        self.voice_active = not self.voice_active
        if self.voice_active:
            self.btn_voice.config(text="MIC ●", fg=TEAL_BRT)
            self._q.put(("state", "listening"))
        else:
            self.btn_voice.config(text="MIC ○", fg=TEXT_DIM)
            self._q.put(("state", "idle"))

    def _reset(self):
        self.assistant.reset()
        self.canvas._chat_lines.clear()

    def _on_close(self):
        self.canvas.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        root.attributes("-alpha", 0.97)
    except Exception:
        pass
    JarvisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
