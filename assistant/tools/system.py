import os
import webbrowser
import datetime
import subprocess
import platform

try:
    import pyautogui
    _PYAUTOGUI = True
except ImportError:
    _PYAUTOGUI = False


_APP_MAP = {
    "chrome": "start chrome",
    "google chrome": "start chrome",
    "firefox": "start firefox",
    "edge": "start msedge",
    "notepad": "start notepad",
    "calculator": "start calc",
    "calendar": "start outlookcal:",
    "explorer": "start explorer",
    "file explorer": "start explorer",
    "spotify": "start spotify",
    "discord": "start discord",
    "vscode": "start code",
    "visual studio code": "start code",
    "terminal": "start cmd",
    "cmd": "start cmd",
    "powershell": "start powershell",
    "word": "start winword",
    "excel": "start excel",
    "outlook": "start outlook",
    "paint": "start mspaint",
    "task manager": "start taskmgr",
}


def get_time() -> str:
    return datetime.datetime.now().strftime("It's %I:%M %p")


def get_date() -> str:
    return datetime.datetime.now().strftime("Today is %A, %B %d, %Y")


def take_screenshot(filename: str | None = None) -> str:
    if not _PYAUTOGUI:
        return "PyAutoGUI not installed. Run: pip install pyautogui"
    if not filename:
        filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    try:
        img = pyautogui.screenshot()
        img.save(filename)
        return f"Screenshot saved as {filename}"
    except Exception as e:
        return f"Screenshot failed: {e}"


def open_application(app_name: str) -> str:
    cmd = _APP_MAP.get(app_name.lower().strip())
    try:
        if cmd:
            os.system(cmd)
        else:
            os.system(f"start {app_name}")
        return f"Opening {app_name}."
    except Exception as e:
        return f"Could not open {app_name}: {e}"


def open_website(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opening {url} in your browser."


def create_file(filename: str, content: str) -> str:
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File '{filename}' created."
    except Exception as e:
        return f"Failed to create '{filename}': {e}"


def read_file(filename: str) -> str:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"File '{filename}' not found."
    except Exception as e:
        return f"Failed to read '{filename}': {e}"


def run_command(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=15
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if result.returncode != 0:
            return f"Command failed (exit {result.returncode}): {err or out}"
        return out or "Command executed."
    except subprocess.TimeoutExpired:
        return "Command timed out."
    except Exception as e:
        return f"Command error: {e}"
