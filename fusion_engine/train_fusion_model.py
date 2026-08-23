"""
=====================================================
XGBoost Multimodal Fusion Model Trainer
=====================================================

Purpose:
Train final fusion model using shared feature rows.

Input:

fusion_live_dataset01.csv

fusion_dataset/fusion_live_dataset.csv

Output:
fusion_engine/model/xgboost_fusion_model.pkl
fusion_engine/model/label_encoder.pkl
fusion_engine/model/feature_columns.pkl
fusion_engine/model/training_report.txt
fusion_engine/model/confusion_matrix.csv
fusion_engine/model/feature_importance.csv

Run:
    python fusion_engine/train_fusion_model.py

=====================================================
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FUSION_DATASET_DIR = os.path.join(
    BASE_DIR,
    "fusion_dataset"
)

IDEAL_DATASET_PATH = os.path.join(
    FUSION_DATASET_DIR,
    "fusion_live_dataset01.csv"
)

LIVE_DATASET_PATH = os.path.join(
    FUSION_DATASET_DIR,
    "fusion_live_dataset.csv"
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

REPORT_PATH = os.path.join(
    MODEL_DIR,
    "training_report.txt"
)

CONFUSION_MATRIX_PATH = os.path.join(
    MODEL_DIR,
    "confusion_matrix.csv"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    MODEL_DIR,
    "feature_importance.csv"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================================
# Feature Columns
# =====================================================

FEATURE_COLUMNS = [
    "face_detected",
    "head_away",
    "eyes_closed_time",
    "head_away_time",
    "no_face_time",
    "drowsy",
    "phone_detected",

    "keyboard_active",
    "mouse_active",

    "app_focused",
    "app_distracted",
    "app_neutral",
    "app_unknown",
    "app_confidence",

    "speech_status",
    "audio_study_probability",
]

TARGET_COLUMN = "final_label"


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
# Dataset Loading
# =====================================================

def select_dataset_path():
    """
    Selects the best available dataset.
    Priority:
    1. fusion_live_dataset01.csv
    2. fusion_live_dataset.csv
    """

    if os.path.exists(IDEAL_DATASET_PATH):
        return IDEAL_DATASET_PATH

    if os.path.exists(LIVE_DATASET_PATH):
        return LIVE_DATASET_PATH

    print("\nERROR: No fusion dataset found.")
    print("Expected one of these files:")
    print(IDEAL_DATASET_PATH)
    print(LIVE_DATASET_PATH)
    sys.exit(1)


def load_dataset(dataset_path):
    print("=" * 70)
    print("LOADING FUSION DATASET")
    print("=" * 70)
    print("Dataset:", dataset_path)

    df = pd.read_csv(
        dataset_path,
        encoding="utf-8"
    )

    print("Original Rows   :", len(df))
    print("Original Columns:", len(df.columns))

    return df


# =====================================================
# Dataset Cleaning
# =====================================================

def normalize_label(label):
    """
    Converts labels into standard format:
    Focused, Distracted, Neutral, Absent
    """

    if pd.isna(label):
        return ""

    label = str(label).strip().lower()

    mapping = {
        "focused": "Focused",
        "focus": "Focused",

        "distracted": "Distracted",
        "distraction": "Distracted",

        "neutral": "Neutral",

        "absent": "Absent",
        "no face": "Absent",
        "away": "Absent",
    }

    return mapping.get(label, "")


def clean_dataset(df):
    print("\n" + "=" * 70)
    print("CLEANING DATASET")
    print("=" * 70)

    # ---------------------------------------------
    # Check target column
    # ---------------------------------------------
    if TARGET_COLUMN not in df.columns:
        print(f"ERROR: Target column '{TARGET_COLUMN}' not found.")
        print("Available columns:")
        print(list(df.columns))
        sys.exit(1)

    # ---------------------------------------------
    # Normalize labels
    # ---------------------------------------------
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(
        normalize_label
    )

    # Remove rows without label
    df = df[df[TARGET_COLUMN] != ""].copy()

    # ---------------------------------------------
    # Add missing feature columns safely
    # ---------------------------------------------
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            default_value = DEFAULT_VALUES[column]
            df[column] = default_value
            print(
                f"Missing column added: {column} = {default_value}"
            )

    # ---------------------------------------------
    # Convert features to numeric
    # ---------------------------------------------
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        df[column] = df[column].fillna(
            DEFAULT_VALUES[column]
        )

    # ---------------------------------------------
    # Remove impossible duplicate header rows
    # ---------------------------------------------
    df = df[df[TARGET_COLUMN].isin(
        ["Focused", "Distracted", "Neutral", "Absent"]
    )].copy()

    # ---------------------------------------------
    # Final safety checks
    # ---------------------------------------------
    null_count = df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum().sum()

    print("Cleaned Rows:", len(df))
    print("Null Values :", null_count)

    if len(df) == 0:
        print("ERROR: Dataset is empty after cleaning.")
        sys.exit(1)

    print("\nClass Distribution:")
    print(df[TARGET_COLUMN].value_counts())

    return df


# =====================================================
# Train Model
# =====================================================

def train_model(df):
    print("\n" + "=" * 70)
    print("TRAINING XGBOOST FUSION MODEL")
    print("=" * 70)

    X = df[FEATURE_COLUMNS].copy()
    y_text = df[TARGET_COLUMN].copy()

    # ---------------------------------------------
    # Label Encoding
    # ---------------------------------------------
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    print("\nLabels:")
    for index, label in enumerate(label_encoder.classes_):
        print(f"{index} -> {label}")

    if len(label_encoder.classes_) < 2:
        print("ERROR: Need at least 2 classes to train.")
        sys.exit(1)

    # ---------------------------------------------
    # Train/Test Split
    # ---------------------------------------------
    class_counts = y_text.value_counts()
    can_stratify = class_counts.min() >= 2

    if can_stratify:
        stratify_value = y
    else:
        stratify_value = None
        print("\nWARNING: Stratified split disabled because one class has too few rows.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_value
    )

    print("\nTraining Rows:", len(X_train))
    print("Testing Rows :", len(X_test))

    # ---------------------------------------------
    # Sample Weights
    # ---------------------------------------------
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    # ---------------------------------------------
    # XGBoost Model
    # ---------------------------------------------
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),

        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,

        subsample=0.90,
        colsample_bytree=0.90,

        eval_metric="mlogloss",
        random_state=42,

        tree_method="hist",
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )

    # ---------------------------------------------
    # Evaluation
    # ---------------------------------------------
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)

    # ---------------------------------------------
    # Feature Importance
    # ---------------------------------------------
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    print("\nTop Feature Importance:")
    print(importance_df.head(15))

    # ---------------------------------------------
    # Save Artifacts
    # ---------------------------------------------
    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        label_encoder,
        LABEL_ENCODER_PATH
    )

    joblib.dump(
        FEATURE_COLUMNS,
        FEATURE_COLUMNS_PATH
    )

    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )

    cm_df.to_csv(
        CONFUSION_MATRIX_PATH,
        encoding="utf-8"
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
        encoding="utf-8"
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        f.write("XGBoost Multimodal Fusion Model Training Report\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Dataset: {select_dataset_path()}\n")
        f.write(f"Total Rows: {len(df)}\n")
        f.write(f"Training Rows: {len(X_train)}\n")
        f.write(f"Testing Rows: {len(X_test)}\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")

        f.write("Labels:\n")
        for index, label in enumerate(label_encoder.classes_):
            f.write(f"{index} -> {label}\n")

        f.write("\nClass Distribution:\n")
        f.write(str(df[TARGET_COLUMN].value_counts()))
        f.write("\n\n")

        f.write("Feature Columns:\n")
        for column in FEATURE_COLUMNS:
            f.write(f"- {column}\n")

        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm_df))
        f.write("\n\nFeature Importance:\n")
        f.write(str(importance_df))

    print("\n" + "=" * 70)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 70)
    print("Model             :", MODEL_PATH)
    print("Label Encoder     :", LABEL_ENCODER_PATH)
    print("Feature Columns   :", FEATURE_COLUMNS_PATH)
    print("Training Report   :", REPORT_PATH)
    print("Confusion Matrix  :", CONFUSION_MATRIX_PATH)
    print("Feature Importance:", FEATURE_IMPORTANCE_PATH)

    return model, label_encoder


# =====================================================
# Main
# =====================================================

def main():
    dataset_path = select_dataset_path()
    df = load_dataset(dataset_path)
    df = clean_dataset(df)
    train_model(df)


if __name__ == "__main__":
    main()