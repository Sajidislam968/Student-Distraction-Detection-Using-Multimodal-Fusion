"""
=====================================================
Test Fusion Predictor
=====================================================

Run:
    python fusion_engine/test_fusion_predictor.py
=====================================================
"""

from fusion_predictor import FusionPredictor


def main():
    predictor = FusionPredictor()

    sample_snapshot = {
        "face_detected": 1,
        "head_away": 0,
        "eyes_closed_time": 0.0,
        "head_away_time": 0.0,
        "no_face_time": 0.0,
        "drowsy": 0,
        "phone_detected": 0,

        "keyboard_active": 1,
        "mouse_active": 1,

        "app_label": "Focused",
        "app_confidence": 92.5,

        "speech_status": 0,
        "study_probability": 75.0,
    }

    result = predictor.predict(sample_snapshot)

    print("=" * 70)
    print("FUSION PREDICTOR TEST")
    print("=" * 70)

    print("Prediction :", result["fusion_prediction"])
    print("Confidence :", result["fusion_confidence"], "%")

    print("\nProbabilities:")
    for label, probability in result["fusion_probabilities"].items():
        print(f"{label:<12}: {probability:>6.2f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()