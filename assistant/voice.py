import pyttsx3
import speech_recognition as sr


class VoiceIO:
    def __init__(self):
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty("voices")
        # prefer female voice (index 1) if available
        if len(voices) > 1:
            self.engine.setProperty("voice", voices[1].id)
        self.engine.setProperty("rate", 165)
        self.engine.setProperty("volume", 1.0)

        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.0
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

    def speak(self, text: str):
        # Strip markdown for cleaner TTS output
        clean = (
            text.replace("**", "").replace("*", "").replace("#", "")
            .replace("`", "").replace("_", " ")
        )
        print(f"\nJarvis: {clean}\n")
        self.engine.say(clean)
        self.engine.runAndWait()

    def listen(self, timeout: int = 8, phrase_limit: int = 20) -> str | None:
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
            print("Listening... ", end="", flush=True)
            try:
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_limit
                )
                text = self.recognizer.recognize_google(audio)
                print(f"\nYou: {text}")
                return text
            except sr.WaitTimeoutError:
                print("(timeout)")
                return None
            except sr.UnknownValueError:
                print("(unclear)")
                return None
            except sr.RequestError as e:
                print(f"\nSTT service error: {e}")
                return None
