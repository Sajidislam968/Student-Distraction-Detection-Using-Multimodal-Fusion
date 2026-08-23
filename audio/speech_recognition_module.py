import numpy as np
import sounddevice as sd
import speech_recognition as sr


class SpeechRecognitionModule:

    def __init__(self, device_index=7, samplerate=16000):

        self.device_index = device_index
        self.samplerate = samplerate
        self.recognizer = sr.Recognizer()


    def listen_and_convert(self, duration=5):

        try:
            print(f"Recording for {duration} seconds... Speak now.")

            audio_data = sd.rec(
                int(duration * self.samplerate),
                samplerate=self.samplerate,
                channels=1,
                dtype="int16",
                device=self.device_index
            )

            sd.wait()

            print("Recording finished. Converting speech to text...")

            audio_bytes = audio_data.tobytes()

            audio = sr.AudioData(
                audio_bytes,
                self.samplerate,
                2
            )

            try:
                text = self.recognizer.recognize_google(audio)
                return text

            except sr.UnknownValueError:
                return ""

            except sr.RequestError:
                return "Speech recognition service error"

        except Exception as e:
            return f"Microphone error: {e}"