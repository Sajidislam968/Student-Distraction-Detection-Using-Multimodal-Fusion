"""
=====================================================
Fusion Predictor
=====================================================

Loads trained XGBoost fusion model and predicts final
student state from the current shared_state snapshot.

Outputs:
Focused / Distracted / Neutral / Absent

=====================================================
"""

import os
import joblib
import pandas as pd


# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "fusion_engine",
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_fusion_model.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)


# =====================================================
# Default Values
# =====================================================

DEFAULT_VALUES = {
    "face_detected": 0,
    "head_away": 0,
    "eyes_closed_time": 0.0,
    "head_away_time": 0.0,
    "no_face_time": 0.0,
    "drowsy": 0,
    "phone_detected": 0,

    "keyboard_active": 0,
    "mouse_active": 0,

    "app_focused": 0,
    "app_distracted": 0,
    "app_neutral": 0,
    "app_unknown": 1,
    "app_confidence": 0.0,

    "speech_status": 0,
    "audio_study_probability": 50.0,
}


# =====================================================
# Fusion Predictor Class
# =====================================================

class FusionPredictor:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Fusion model not found: {MODEL_PATH}"
            )

        if not os.path.exists(LABEL_ENCODER_PATH):
            raise FileNotFoundError(
                f"Label encoder not found: {LABEL_ENCODER_PATH}"
            )

        if not os.path.exists(FEATURE_COLUMNS_PATH):
            raise FileNotFoundError(
                f"Feature columns file not found: {FEATURE_COLUMNS_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        self.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    def snapshot_to_features(self, snapshot):
        """
        Converts shared_state snapshot into one numeric feature row.
        """

        app_label = str(
            snapshot.get("app_label", "Unknown")
        )

        feature_dict = {
            "face_detected": snapshot.get("face_detected", 0),
            "head_away": snapshot.get("head_away", 0),
            "eyes_closed_time": snapshot.get("eyes_closed_time", 0.0),
            "head_away_time": snapshot.get("head_away_time", 0.0),
            "no_face_time": snapshot.get("no_face_time", 0.0),
            "drowsy": snapshot.get("drowsy", 0),
            "phone_detected": snapshot.get("phone_detected", 0),

            "keyboard_active": snapshot.get("keyboard_active", 0),
            "mouse_active": snapshot.get("mouse_active", 0),

            "app_focused": 1 if app_label == "Focused" else 0,
            "app_distracted": 1 if app_label == "Distracted" else 0,
            "app_neutral": 1 if app_label == "Neutral" else 0,
            "app_unknown": 1 if app_label == "Unknown" else 0,
            "app_confidence": snapshot.get("app_confidence", 0.0),

            "speech_status": snapshot.get("speech_status", 0),
            "audio_study_probability": snapshot.get("study_probability", 50.0),
        }

        clean_row = {}

        for column in self.feature_columns:
            value = feature_dict.get(
                column,
                DEFAULT_VALUES.get(column, 0)
            )

            try:
                value = float(value)
            except Exception:
                value = DEFAULT_VALUES.get(column, 0)

            clean_row[column] = value

        X = pd.DataFrame(
            [clean_row],
            columns=self.feature_columns
        )

        return X

    def predict(self, snapshot):
        """
        Returns:
        {
            "fusion_prediction": "Focused",
            "fusion_confidence": 97.32,
            "fusion_probabilities": {
                "Absent": 0.01,
                "Distracted": 1.22,
                "Focused": 97.32,
                "Neutral": 1.45
            }
        }
        """

        X = self.snapshot_to_features(snapshot)

        pred_encoded = self.model.predict(X)[0]
        prediction = self.label_encoder.inverse_transform(
            [pred_encoded]
        )[0]

        probabilities_raw = self.model.predict_proba(X)[0]

        probabilities = {}

        for label, prob in zip(
            self.label_encoder.classes_,
            probabilities_raw
        ):
            probabilities[label] = round(float(prob) * 100, 2)

        confidence = max(probabilities.values())

        return {
            "fusion_prediction": prediction,
            "fusion_confidence": confidence,
            "fusion_probabilities": probabilities,
            "fusion_features": X.iloc[0].to_dict()
        }