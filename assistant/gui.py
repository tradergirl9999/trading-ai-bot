"""
Jarvis GUI — dark trading-dashboard style
Run: python gui.py
"""

import os
import sys
import threading
import time
import math
import queue
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

if not os.getenv("ANTHROPIC_API_KEY"):
    sys.exit("Error: ANTHROPIC_API_KEY not set. Add it to assistant/.env")

import tkinter as tk
from tkinter import ttk, scrolledtext

from brain import JarvisAssistant
from voice import VoiceIO
from tools import reminders

# ─── Palette ────────────────────────────────────────────────────────────────
BG         = "#0d0f14"
BG_PANEL   = "#13161e"
BG_INPUT   = "#1a1d27"
BG_BUBBLE_USER    = "#1e2a1e"
BG_BUBBLE_JARVIS  = "#141820"
ACCENT     = "#00e5a0"       # bright green
ACCENT2    = "#00bfff"       # cyan
RED        = "#ff4757"
YELLOW     = "#ffd700"
TEXT       = "#e8ecf0"
TEXT_DIM   = "#5a6070"
TEXT_TIME  = "#3a4050"
FONT_MAIN  = ("Consolas", 11)
FONT_CHAT  = ("Consolas", 11)
FONT_SMALL = ("Consolas", 9)
FONT_TITLE = ("Consolas", 13, "bold")
FONT_STATUS= ("Consolas", 10)

STATES = {
    "idle":      ("STANDBY",   TEXT_DIM, 0.3),
    "listening": ("LISTENING", ACCENT,   1.0),
    "thinking":  ("THINKING",  ACCENT2,  0.7),
    "speaking":  ("SPEAKING",  YELLOW,   0.9),
    "error":     ("ERROR",     RED,      1.0),
}


class PulseCanvas(tk.Canvas):
    """Animated pulse ring that reflects assistant state."""

    def __init__(self, parent, size=100, **kw):
        super().__init__(parent, width=size, height=size,
                         bg=BG_PANEL, highlightthickness=0, **kw)
        self.size = size
        self.cx = size // 2
        self.cy = size // 2
        self.r_base = size // 2 - 12
        self._phase = 0.0
        self._state = "idle"
        self._color = TEXT_DIM
        self._alpha = 0.3
        self._running = True
        self._animate()

    def set_state(self, state: str):
        _, color, alpha = STATES.get(state, STATES["idle"])
        self._state = state
        self._color = color
        self._alpha = alpha

    def _animate(self):
        if not self._running:
            return
        self.delete("all")
        t = self._phase

        if self._state == "listening":
            for i in range(3):
                phase_offset = i * (2 * math.pi / 3)
                radius = self.r_base + 5 * math.sin(t * 2 + phase_offset)
                opacity = int(255 * (0.4 + 0.3 * math.sin(t + phase_offset)))
                color = _blend(self._color, opacity)
                self.create_oval(
                    self.cx - radius, self.cy - radius,
                    self.cx + radius, self.cy + radius,
                    outline=color, width=2
                )
        elif self._state == "thinking":
            segments = 8
            for i in range(segments):
                angle = (t * 3 + i * (2 * math.pi / segments))
                r1 = self.r_base - 4
                r2 = self.r_base + 4
                x1 = self.cx + r1 * math.cos(angle)
                y1 = self.cy + r1 * math.sin(angle)
                x2 = self.cx + r2 * math.cos(angle)
                y2 = self.cy + r2 * math.sin(angle)
                brightness = int(255 * (0.3 + 0.7 * ((i / segments + t / (2 * math.pi)) % 1.0)))
                color = _blend(self._color, brightness)
                self.create_line(x1, y1, x2, y2, fill=color, width=2)
        elif self._state == "speaking":
            bars = 7
            bar_w = 6
            spacing = 4
            total_w = bars * bar_w + (bars - 1) * spacing
            x0 = self.cx - total_w // 2
            for i in range(bars):
                h = 8 + 18 * abs(math.sin(t * 4 + i * 0.8))
                x = x0 + i * (bar_w + spacing)
                opacity = int(200 + 55 * math.sin(t * 3 + i))
                color = _blend(self._color, opacity)
                self.create_rectangle(
                    x, self.cy - h, x + bar_w, self.cy + h,
                    fill=color, outline=""
                )
        else:
            # idle — static ring
            self.create_oval(
                self.cx - self.r_base, self.cy - self.r_base,
                self.cx + self.r_base, self.cy + self.r_base,
                outline=TEXT_DIM, width=1
            )
            self.create_oval(
                self.cx - 6, self.cy - 6,
                self.cx + 6, self.cy + 6,
                fill=TEXT_DIM, outline=""
            )

        # centre dot
        dot_r = 5
        self.create_oval(
            self.cx - dot_r, self.cy - dot_r,
            self.cx + dot_r, self.cy + dot_r,
            fill=self._color, outline=""
        )

        self._phase += 0.08
        self.after(50, self._animate)

    def stop(self):
        self._running = False


def _blend(hex_color: str, alpha: int) -> str:
    """Return hex color with brightness scaled by alpha (0-255)."""
    alpha = max(0, min(255, alpha))
    r = int(int(hex_color[1:3], 16) * alpha / 255)
    g = int(int(hex_color[3:5], 16) * alpha / 255)
    b = int(int(hex_color[5:7], 16) * alpha / 255)
    return f"#{r:02x}{g:02x}{b:02x}"


class JarvisApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("JARVIS")
        self.root.configure(bg=BG)
        self.root.geometry("820x680")
        self.root.minsize(640, 500)
        self.root.resizable(True, True)

        self.assistant = JarvisAssistant()
        self.voice = VoiceIO()
        self.voice_active = True
        self._q: queue.Queue = queue.Queue()

        reminders.set_speak_callback(self._speak)

        self._build_ui()
        self._set_state("idle")
        self.root.after(100, self._process_queue)
        self.root.after(600, self._start_voice_thread)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG_PANEL, height=52)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)

        tk.Label(top, text="◈  JARVIS", font=FONT_TITLE,
                 bg=BG_PANEL, fg=ACCENT).pack(side="left", padx=18, pady=14)

        self.status_label = tk.Label(top, text="● STANDBY", font=FONT_STATUS,
                                     bg=BG_PANEL, fg=TEXT_DIM)
        self.status_label.pack(side="left", padx=6)

        # reset button
        btn_reset = tk.Button(
            top, text="RESET", font=FONT_SMALL, bg=BG_INPUT, fg=TEXT_DIM,
            activebackground=BG_INPUT, activeforeground=ACCENT,
            relief="flat", bd=0, padx=10, pady=6, cursor="hand2",
            command=self._reset
        )
        btn_reset.pack(side="right", padx=12, pady=10)

        # voice toggle
        self.btn_voice = tk.Button(
            top, text="MIC  ON", font=FONT_SMALL, bg=BG_INPUT, fg=ACCENT,
            activebackground=BG_INPUT, activeforeground=ACCENT,
            relief="flat", bd=0, padx=10, pady=6, cursor="hand2",
            command=self._toggle_voice
        )
        self.btn_voice.pack(side="right", padx=4, pady=10)

        sep = tk.Frame(self.root, bg=ACCENT, height=1)
        sep.pack(fill="x")

        # ── Body: chat + pulse ───────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        # left: chat
        chat_frame = tk.Frame(body, bg=BG)
        chat_frame.pack(side="left", fill="both", expand=True, padx=(14, 6), pady=10)

        self.chat = tk.Text(
            chat_frame,
            bg=BG, fg=TEXT, font=FONT_CHAT,
            relief="flat", bd=0, wrap="word",
            state="disabled", cursor="arrow",
            selectbackground=BG_PANEL,
        )
        self.chat.pack(fill="both", expand=True)

        sb = tk.Scrollbar(chat_frame, command=self.chat.yview,
                          bg=BG_PANEL, troughcolor=BG, relief="flat", bd=0)
        sb.pack(side="right", fill="y")
        self.chat.configure(yscrollcommand=sb.set)

        self._configure_tags()

        # right: pulse + status panel
        right = tk.Frame(body, bg=BG_PANEL, width=160)
        right.pack(side="right", fill="y", padx=(0, 10), pady=10)
        right.pack_propagate(False)

        self.pulse = PulseCanvas(right, size=110)
        self.pulse.pack(pady=(24, 8))

        self.state_text = tk.Label(right, text="STANDBY", font=FONT_SMALL,
                                   bg=BG_PANEL, fg=TEXT_DIM)
        self.state_text.pack()

        sep2 = tk.Frame(right, bg=BG_INPUT, height=1)
        sep2.pack(fill="x", padx=12, pady=14)

        tk.Label(right, text="QUICK", font=FONT_SMALL,
                 bg=BG_PANEL, fg=TEXT_DIM).pack()
        for label, prompt in [
            ("Market", "Give me a quick market overview"),
            ("BTC",    "What is Bitcoin doing right now?"),
            ("News",   "What's in the financial news today?"),
            ("Weather","What's the weather like?"),
        ]:
            b = tk.Button(
                right, text=label, font=FONT_SMALL, bg=BG_INPUT, fg=TEXT_DIM,
                activebackground=BG, activeforeground=ACCENT,
                relief="flat", bd=0, padx=8, pady=5, cursor="hand2",
                command=lambda p=prompt: self._submit_text(p)
            )
            b.pack(fill="x", padx=12, pady=2)

        # ── Bottom input bar ──────────────────────────────────────────────────
        bottom_sep = tk.Frame(self.root, bg=BG_INPUT, height=1)
        bottom_sep.pack(fill="x")

        bottom = tk.Frame(self.root, bg=BG_PANEL, height=52)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)

        self.entry = tk.Entry(
            bottom, bg=BG_INPUT, fg=TEXT, font=FONT_CHAT,
            relief="flat", bd=0, insertbackground=ACCENT,
            disabledbackground=BG_INPUT,
        )
        self.entry.pack(side="left", fill="both", expand=True, padx=14, pady=12, ipady=4)
        self.entry.bind("<Return>", self._on_enter)

        send_btn = tk.Button(
            bottom, text="SEND ▶", font=FONT_SMALL, bg=ACCENT, fg=BG,
            activebackground=ACCENT2, activeforeground=BG,
            relief="flat", bd=0, padx=14, pady=8, cursor="hand2",
            command=lambda: self._on_enter(None)
        )
        send_btn.pack(side="right", padx=10, pady=10)

    def _configure_tags(self):
        self.chat.tag_configure("user_name",   foreground=ACCENT,  font=("Consolas", 9, "bold"))
        self.chat.tag_configure("user_bubble", foreground=TEXT,     font=FONT_CHAT,
                                background=BG_BUBBLE_USER, lmargin1=8, lmargin2=8, rmargin=8)
        self.chat.tag_configure("jarvis_name", foreground=ACCENT2, font=("Consolas", 9, "bold"))
        self.chat.tag_configure("jarvis_bubble", foreground=TEXT,   font=FONT_CHAT,
                                background=BG_BUBBLE_JARVIS, lmargin1=8, lmargin2=8, rmargin=8)
        self.chat.tag_configure("time",        foreground=TEXT_TIME, font=FONT_SMALL)
        self.chat.tag_configure("dim",         foreground=TEXT_DIM,  font=FONT_SMALL)
        self.chat.tag_configure("spacer",      font=("Consolas", 4))

    # ── State management ──────────────────────────────────────────────────────

    def _set_state(self, state: str):
        label, color, _ = STATES.get(state, STATES["idle"])
        self.pulse.set_state(state)
        dot = "◉" if state != "idle" else "●"
        self.status_label.config(text=f"{dot} {label}", fg=color)
        self.state_text.config(text=label, fg=color)

    # ── Chat display ──────────────────────────────────────────────────────────

    def _append_message(self, sender: str, text: str):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M")

        self.chat.config(state="normal")
        self.chat.insert("end", "\n", "spacer")

        if sender == "you":
            self.chat.insert("end", f"  YOU  {ts}\n", "user_name")
            self.chat.insert("end", f"  {text}\n", "user_bubble")
        else:
            self.chat.insert("end", f"  JARVIS  {ts}\n", "jarvis_name")
            # Indent each line of response
            for line in text.split("\n"):
                self.chat.insert("end", f"  {line}\n", "jarvis_bubble")

        self.chat.config(state="disabled")
        self.chat.see("end")

    def _append_dim(self, text: str):
        self.chat.config(state="normal")
        self.chat.insert("end", f"\n  {text}\n", "dim")
        self.chat.config(state="disabled")
        self.chat.see("end")

    # ── Queue (thread-safe UI updates) ───────────────────────────────────────

    def _process_queue(self):
        while not self._q.empty():
            cmd, *args = self._q.get_nowait()
            if cmd == "state":
                self._set_state(args[0])
            elif cmd == "message":
                self._append_message(*args)
            elif cmd == "dim":
                self._append_dim(args[0])
        self.root.after(60, self._process_queue)

    def _q_put(self, *args):
        self._q.put(args)

    # ── Voice loop ───────────────────────────────────────────────────────────

    def _start_voice_thread(self):
        self._append_dim("Jarvis online. Listening for your voice or use the text box below.")
        t = threading.Thread(target=self._voice_loop, daemon=True)
        t.start()

    def _voice_loop(self):
        while True:
            if not self.voice_active:
                time.sleep(0.5)
                continue
            self._q_put("state", "listening")
            text = self.voice.listen()
            if text:
                self._handle_input(text)
            else:
                self._q_put("state", "idle")

    def _handle_input(self, text: str):
        cmd = text.lower().strip()

        if cmd in ("quit", "exit", "goodbye", "bye"):
            self._q_put("message", "you", text)
            self._q_put("state", "speaking")
            self._speak("Goodbye. Stay sharp.")
            self._q_put("state", "idle")
            return

        if cmd == "reset":
            self._reset()
            return

        self._q_put("message", "you", text)
        self._q_put("state", "thinking")

        def _run():
            try:
                response = self.assistant.process(text)
                self._q_put("message", "jarvis", response)
                self._q_put("state", "speaking")
                self._speak(response)
            except Exception as e:
                self._q_put("message", "jarvis", f"Error: {e}")
                self._q_put("state", "error")
                time.sleep(1)
            finally:
                self._q_put("state", "listening" if self.voice_active else "idle")

        threading.Thread(target=_run, daemon=True).start()

    def _speak(self, text: str):
        self.voice.speak(text)

    # ── Text input ────────────────────────────────────────────────────────────

    def _on_enter(self, event):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        threading.Thread(target=self._handle_input, args=(text,), daemon=True).start()

    def _submit_text(self, text: str):
        threading.Thread(target=self._handle_input, args=(text,), daemon=True).start()

    # ── Controls ──────────────────────────────────────────────────────────────

    def _toggle_voice(self):
        self.voice_active = not self.voice_active
        if self.voice_active:
            self.btn_voice.config(text="MIC  ON", fg=ACCENT)
            self._q_put("state", "listening")
        else:
            self.btn_voice.config(text="MIC OFF", fg=TEXT_DIM)
            self._q_put("state", "idle")

    def _reset(self):
        self.assistant.reset()
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")
        self._append_dim("Memory cleared. Ready.")

    def _on_close(self):
        self.pulse.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    # Remove default window decorations on Windows for a cleaner look
    try:
        root.attributes("-alpha", 0.97)
    except Exception:
        pass
    app = JarvisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
