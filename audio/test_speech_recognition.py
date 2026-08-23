from speech_recognition_module import SpeechRecognitionModule
from study_audio_classifier import StudyAudioClassifier


speech_recognizer = SpeechRecognitionModule(device_index=7)
classifier = StudyAudioClassifier()


while True:

    input("Press ENTER and speak for 5 seconds...")

    text = speech_recognizer.listen_and_convert(duration=5)

    print("\nRecognized Text:")
    print(text)

    result = classifier.classify(text)

    print("\nAudio Classification")
    print("--------------------")
    print("Label:", result["label"])
    print("Study Probability:", result["study_probability"], "%")
    print("Study Keywords:", result["study_keywords"])
    print("Non-study Keywords:", result["non_study_keywords"])

    print("\n====================================\n")