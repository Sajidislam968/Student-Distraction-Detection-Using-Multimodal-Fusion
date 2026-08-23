import csv
import os
from datetime import datetime


class AudioLogger:

    def __init__(self):

        os.makedirs("logs", exist_ok=True)

        self.file = "logs/audio_log.csv"

        if not os.path.exists(self.file):

            with open(self.file, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow(
                    [
                        "Time",
                        "Volume",
                        "Noise Level",
                        "Speech Detected",
                        "Speech Duration",
                        "Silence Duration",
                        "Audio Status"
                    ]
                )


    def log(
        self,
        volume,
        noise_level,
        speech_detected,
        speech_duration,
        silence_duration,
        audio_status
    ):

        with open(self.file, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    round(volume, 2),
                    noise_level,
                    speech_detected,
                    speech_duration,
                    silence_duration,
                    audio_status
                ]
            )