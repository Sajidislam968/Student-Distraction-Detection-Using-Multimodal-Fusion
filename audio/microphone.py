import sounddevice as sd
import numpy as np


class Microphone:

    def __init__(self, device=None, samplerate=16000):

        self.device = device
        self.samplerate = samplerate


    def record_chunk(self, duration=1):

        audio = sd.rec(
            int(duration * self.samplerate),
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            device=self.device
        )

        sd.wait()

        return audio.flatten()



if __name__ == "__main__":

    mic = Microphone()

    print("Recording 5 seconds...")

    data = mic.record_chunk(5)

    print("Audio captured")
    print("Samples:", len(data))