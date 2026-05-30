"""
Jarvis — Personal AI Assistant
--------------------------------
Run:  python main.py
Deps: pip install -r requirements.txt
Env:  ANTHROPIC_API_KEY in .env or system environment
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

if not os.getenv("ANTHROPIC_API_KEY"):
    sys.exit("Error: ANTHROPIC_API_KEY not set. Add it to assistant/.env")

from brain import JarvisAssistant
from voice import VoiceIO
from tools import reminders

BANNER = """
╔══════════════════════════════════════╗
║        JARVIS  —  AI Assistant       ║
║  Say 'quit' or Ctrl+C to exit        ║
║  Say 'reset' to clear memory         ║
║  Say 'type' for keyboard input       ║
╚══════════════════════════════════════╝
"""

EXIT_WORDS = {"quit", "exit", "goodbye", "bye", "stop", "shutdown"}


def main():
    print(BANNER)
    voice = VoiceIO()
    assistant = JarvisAssistant()

    # Wire reminder alerts to the TTS engine
    reminders.set_speak_callback(voice.speak)

    voice.speak("Hello! Jarvis online. How can I help you?")

    use_voice = True

    while True:
        try:
            if use_voice:
                user_input = voice.listen()
                if not user_input:
                    continue
            else:
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    break
                if not user_input:
                    continue

            cmd = user_input.lower().strip()

            if cmd in EXIT_WORDS:
                voice.speak("Goodbye. Stay sharp.")
                break

            if cmd == "reset":
                assistant.reset()
                voice.speak("Memory cleared. Fresh start.")
                continue

            if cmd == "type":
                use_voice = False
                print("Switched to keyboard input. Type 'voice' to go back.\n")
                continue

            if cmd == "voice":
                use_voice = True
                voice.speak("Voice mode on.")
                continue

            if cmd == "reminders":
                result = reminders.list_reminders()
                voice.speak(result)
                continue

            response = assistant.process(user_input)
            if response:
                voice.speak(response)

        except KeyboardInterrupt:
            print()
            voice.speak("Goodbye.")
            break
        except Exception as e:
            print(f"[error] {e}")
            voice.speak("Something went wrong. Try again.")


if __name__ == "__main__":
    main()
