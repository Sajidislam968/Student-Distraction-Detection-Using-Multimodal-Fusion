from microphone import Microphone
from noise_detector import NoiseDetector
from speech_detector import SpeechDetector
from audio_logger import AudioLogger


mic = Microphone(device=7)

noise = NoiseDetector()

speech_detector = SpeechDetector(
    threshold=2.2,
    start_confirm_count=2,
    stop_confirm_count=2
)

logger = AudioLogger()


print("Audio monitoring started")


while True:

    audio = mic.record_chunk(1)

    volume = noise.calculate_volume(audio)

    noise_level = noise.classify_noise(volume)

    speech_info = speech_detector.detect(volume)

    audio_status = speech_detector.get_audio_status(
        speech_info,
        noise_level
    )

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
        audio_status
    )

    logger.log(
    volume,
    noise_level,
    speech_info["speech_detected"],
    speech_info["speech_duration"],
    speech_info["silence_duration"],
    audio_status
)