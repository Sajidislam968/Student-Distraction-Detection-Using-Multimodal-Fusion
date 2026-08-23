import os
import re
import string
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from preprocessing import prepare_text

# =====================================================
# Paths
# =====================================================

DATASET = "app_classifier/dataset/app_titles.csv"

MODEL_DIR = "app_classifier/model"

MODEL_FILE = os.path.join(MODEL_DIR, "app_classifier.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# Text Cleaning
# =====================================================

def clean_text(text):

    text = str(text).lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =====================================================
# Load Dataset
# =====================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(DATASET, encoding="cp1252")

# Remove missing labels
df = df.dropna(subset=["label"])

# Remove blank labels
df = df[df["label"].astype(str).str.strip() != ""]

print(f"Training Samples : {len(df)}")

print("\nClass Distribution")

print(df["label"].value_counts())

# =====================================================
# Build Input Text
# =====================================================

df["text"] = (
    df["process_name"].fillna("").astype(str)
    + " "
    + df["window_title"].fillna("").astype(str)
)

df["text"] = df.apply(
    lambda row: prepare_text(
        row["process_name"],
        row["window_title"]
    ),
    axis=1
)

X = df["text"]
y = df["label"]

# =====================================================
# Train / Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining :", len(X_train))
print("Testing  :", len(X_test))

# =====================================================
# TF-IDF
# =====================================================

vectorizer = TfidfVectorizer(
    lowercase=False,      # already cleaned
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
)

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

print("\nVocabulary Size :", len(vectorizer.vocabulary_))

# =====================================================
# Train Model
# =====================================================

print("\nTraining Model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_train_vec, y_train)

# =====================================================
# Prediction
# =====================================================

pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, pred)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy*100:.2f}%")

print("\nClassification Report\n")

print(classification_report(y_test, pred))

print("Confusion Matrix\n")

cm = confusion_matrix(y_test, pred)

print(cm)

# =====================================================
# Top Features
# =====================================================

print("\nTop Learned Words")

feature_names = vectorizer.get_feature_names_out()

classes = model.classes_

for i, label in enumerate(classes):

    print("\n" + "-" * 40)
    print(label)
    print("-" * 40)

    top = model.coef_[i].argsort()[-15:][::-1]

    for idx in top:

        print(feature_names[idx])

# =====================================================
# Save
# =====================================================

joblib.dump(model, MODEL_FILE)

joblib.dump(vectorizer, VECTORIZER_FILE)

print("\n" + "=" * 60)

print("Model Saved Successfully")

print("=" * 60)

print(MODEL_FILE)

print(VECTORIZER_FILE)