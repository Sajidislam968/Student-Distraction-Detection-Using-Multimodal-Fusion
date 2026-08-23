import time


class SpeechDetector:

    def __init__(
        self,
        threshold=2.2,
        start_confirm_count=2,
        stop_confirm_count=2
    ):

        self.threshold = threshold

        self.start_confirm_count = start_confirm_count
        self.stop_confirm_count = stop_confirm_count

        self.speech_counter = 0
        self.silence_counter = 0

        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None

        self.speech_duration = 0
        self.silence_duration = 0


    def detect(self, volume):

        current_time = time.time()

        above_threshold = volume > self.threshold

        if above_threshold:
            self.speech_counter += 1
            self.silence_counter = 0
        else:
            self.silence_counter += 1
            self.speech_counter = 0


        if not self.is_speaking and self.speech_counter >= self.start_confirm_count:
            self.is_speaking = True
            self.speech_start_time = current_time


        if self.is_speaking and self.silence_counter >= self.stop_confirm_count:
            self.is_speaking = False
            self.last_speech_time = current_time
            self.speech_duration = 0


        if self.is_speaking:
            self.speech_duration = current_time - self.speech_start_time
            self.silence_duration = 0
            self.last_speech_time = current_time

        else:
            self.speech_duration = 0

            if self.last_speech_time is not None:
                self.silence_duration = current_time - self.last_speech_time


        return {
            "speech_detected": self.is_speaking,
            "speech_duration": round(self.speech_duration, 2),
            "silence_duration": round(self.silence_duration, 2)
        }


    def get_audio_status(self, speech_info, noise_level):

        speech_detected = speech_info["speech_detected"]
        speech_duration = speech_info["speech_duration"]

        if noise_level == "Loud" and not speech_detected:
            return "Background Noise"

        if speech_detected and speech_duration < 5:
            return "Short Speech"

        elif speech_detected and speech_duration < 15:
            return "Speaking"

        elif speech_detected and speech_duration >= 15:
            return "Long Conversation"

        else:
            return "Silent"