import numpy as np


class NoiseDetector:

    def calculate_volume(self, audio):
        rms = np.sqrt(np.mean(np.square(audio)))
        volume = rms * 100
        return volume

    def classify_noise(self, volume):

        if volume < 1.0:
            return "Quiet"

        elif volume < 6:
            return "Normal"

        else:
            return "Loud"