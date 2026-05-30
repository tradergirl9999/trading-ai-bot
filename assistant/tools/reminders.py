import threading
import time
from datetime import datetime
from typing import Callable


_active: list[dict] = []
_speak_fn: Callable[[str], None] | None = None


def set_speak_callback(fn: Callable[[str], None]):
    global _speak_fn
    _speak_fn = fn


def set_reminder(message: str, minutes: int) -> str:
    entry = {
        "message": message,
        "minutes": minutes,
        "due_at": datetime.now().timestamp() + minutes * 60,
        "fired": False,
    }
    _active.append(entry)

    def _fire():
        time.sleep(minutes * 60)
        entry["fired"] = True
        alert = f"Reminder: {message}"
        print(f"\n🔔 {alert}")
        if _speak_fn:
            _speak_fn(alert)

    threading.Thread(target=_fire, daemon=True).start()
    return f"Reminder set: '{message}' in {minutes} minute{'s' if minutes != 1 else ''}."


def list_reminders() -> str:
    pending = [r for r in _active if not r["fired"]]
    if not pending:
        return "No pending reminders."
    now = datetime.now().timestamp()
    lines = ["Pending reminders:"]
    for r in pending:
        remaining = max(0, int((r["due_at"] - now) / 60))
        lines.append(f"  • '{r['message']}' — {remaining} min remaining")
    return "\n".join(lines)
