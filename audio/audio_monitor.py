import time

from microphone import Microphone
from noise_detector import NoiseDetector
from speech_detector import SpeechDetector
from audio_logger import AudioLogger

from speech_recognition_module import SpeechRecognitionModule
from semantic_audio_classifier import SemanticAudioClassifier
from study_audio_classifier import StudyAudioClassifier


class AudioMonitor:

    def __init__(self, device_index=7):

        self.device_index = device_index

        self.mic = Microphone(device=device_index)
        self.noise = NoiseDetector()

        self.speech_detector = SpeechDetector(
            threshold=2.2,
            start_confirm_count=2,
            stop_confirm_count=2
        )

        self.logger = AudioLogger()

        self.speech_recognizer = SpeechRecognitionModule(
            device_index=device_index
        )

        self.semantic_classifier = SemanticAudioClassifier()
        self.keyword_classifier = StudyAudioClassifier()

        self.semantic_interval = 30
        self.last_semantic_check = 0

        self.latest_text = ""
        self.latest_audio_label = "Uncertain Audio"
        self.latest_study_probability = 50
        self.latest_method = "None"

        self.running = True


    def run_semantic_analysis(self):

        print("\nSemantic speech analysis started...")

        text = self.speech_recognizer.listen_and_convert(duration=5)

        self.latest_text = text

        try:
            result = self.semantic_classifier.classify(text)
            self.latest_method = "Semantic AI"

        except Exception as e:
            print("Semantic AI failed. Using keyword fallback.")
            print("Error:", e)

            result = self.keyword_classifier.classify(text)
            self.latest_method = "Keyword Fallback"

        self.latest_audio_label = result["label"]
        self.latest_study_probability = result["study_probability"]

        print("\nSemantic Result")
        print("----------------")
        print("Text:", self.latest_text)
        print("Label:", self.latest_audio_label)
        print("Study Probability:", self.latest_study_probability, "%")
        print("Method:", self.latest_method)
        print("----------------\n")


    def start(self):

        print("Final Audio Monitor Started")

        while self.running:

            audio = self.mic.record_chunk(1)

            volume = self.noise.calculate_volume(audio)

            noise_level = self.noise.classify_noise(volume)

            speech_info = self.speech_detector.detect(volume)

            audio_status = self.speech_detector.get_audio_status(
                speech_info,
                noise_level
            )

            current_time = time.time()

            should_run_semantic = (
                speech_info["speech_detected"]
                and speech_info["speech_duration"] >= 5
                and current_time - self.last_semantic_check >= self.semantic_interval
            )

            if should_run_semantic:

                self.last_semantic_check = current_time

                self.run_semantic_analysis()


            print(
                "Volume:",
                round(volume, 2),
                "| Noise:",
                noise_level,
                "| Speech:",
                speech_info["speech_detected"],
                "| Speech Time:",
                speech_info["speech_duration"],
                "sec",
                "| Silence:",
                speech_info["silence_duration"],
                "sec",
                "| Status:",
                audio_status,
                "| Semantic:",
                self.latest_audio_label,
                "| Study:",
                self.latest_study_probability,
                "%",
                "| Method:",
                self.latest_method
            )

            self.logger.log(
                volume,
                noise_level,
                speech_info["speech_detected"],
                speech_info["speech_duration"],
                speech_info["silence_duration"],
                audio_status
            )


    def get_audio_data(self):

        return {
            "speech_status": self.speech_detector.is_speaking,
            "audio_label": self.latest_audio_label,
            "study_probability": self.latest_study_probability,
            "transcript": self.latest_text,
            "method": self.latest_method
        }


if __name__ == "__main__":

    monitor = AudioMonitor(device_index=7)
    monitor.start()