"""
=====================================================
AI Predictor Module
=====================================================

Loads the trained ML model and performs predictions.

Returns:

- Prediction
- Confidence
- Probability Distribution
- Top Contributing Features

=====================================================
"""

import joblib
import numpy as np

from config import (
    MODEL_PATH,
    VECTORIZER_PATH,
    CONFIDENCE_THRESHOLD,
    LABEL_UNKNOWN,
    TOP_FEATURES,
)

from preprocessing import prepare_text


# =====================================================
# Load Model (Only Once)
# =====================================================

print("Loading AI Model...")

model = joblib.load(MODEL_PATH)

vectorizer = joblib.load(VECTORIZER_PATH)

feature_names = vectorizer.get_feature_names_out()

classes = model.classes_

print("AI Model Loaded Successfully.\n")


# =====================================================
# Confidence Bar
# =====================================================

def confidence_bar(confidence, length=20):

    filled = int((confidence / 100) * length)

    return "█" * filled + "░" * (length - filled)


# =====================================================
# Explain Prediction
# =====================================================

def explain_prediction(text, predicted_class):

    X = vectorizer.transform([text])

    row = X.toarray()[0]

    class_index = list(classes).index(predicted_class)

    coef = model.coef_[class_index]

    scores = row * coef

    top_indices = np.argsort(scores)[::-1]

    words = []

    for idx in top_indices:

        if scores[idx] <= 0:
            continue

        words.append(feature_names[idx])

        if len(words) >= TOP_FEATURES:
            break

    return words


# =====================================================
# Predict
# =====================================================

def predict(process_name, window_title):

    text = prepare_text(process_name, window_title)

    X = vectorizer.transform([text])

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    confidence = probabilities.max() * 100

    probability_distribution = {}

    for label, probability in zip(classes, probabilities):

        probability_distribution[label] = probability * 100

    probability_distribution = dict(
        sorted(
            probability_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    if confidence < CONFIDENCE_THRESHOLD:

        prediction = LABEL_UNKNOWN

    features = []

    if prediction != LABEL_UNKNOWN:

        features = explain_prediction(text, prediction)

    return {

        "prediction": prediction,

        "confidence": confidence,

        "bar": confidence_bar(confidence),

        "probabilities": probability_distribution,

        "features": features,

        "text": text

    }