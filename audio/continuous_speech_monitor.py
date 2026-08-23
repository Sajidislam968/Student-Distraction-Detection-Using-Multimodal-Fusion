import time

from speech_recognition_module import SpeechRecognitionModule
from semantic_audio_classifier import SemanticAudioClassifier
from study_audio_classifier import StudyAudioClassifier


speech_recognizer = SpeechRecognitionModule(device_index=7)

semantic_classifier = SemanticAudioClassifier()
keyword_classifier = StudyAudioClassifier()


CHECK_INTERVAL = 5
RECORD_DURATION = 30


print("Continuous speech monitoring started")
print(f"It will record for {RECORD_DURATION} seconds.")
print(f"Then wait {CHECK_INTERVAL} seconds before the next check.")


while True:

    text = speech_recognizer.listen_and_convert(
        duration=RECORD_DURATION
    )

    print("\nRecognized Text:")
    print(text)

    try:
        result = semantic_classifier.classify(text)
        method = "Semantic AI"

    except Exception as e:
        print("Semantic AI failed. Using keyword fallback.")
        print("Error:", e)

        result = keyword_classifier.classify(text)
        method = "Keyword Fallback"


    print("\nAudio Classification")
    print("--------------------")
    print("Method:", method)
    print("Label:", result["label"])
    print("Study Probability:", result["study_probability"], "%")

    if "confidence" in result:
        print("Confidence:", result["confidence"], "%")

    if "raw_label" in result:
        print("Raw Model Label:", result["raw_label"])

    if "study_keywords" in result:
        print("Study Keywords:", result["study_keywords"])

    if "non_study_keywords" in result:
        print("Non-study Keywords:", result["non_study_keywords"])

    print("\nWaiting for next check...")
    print("====================================\n")

    time.sleep(CHECK_INTERVAL)