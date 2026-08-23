"""
=====================================================
MODEL COMPARISON SCRIPT
=====================================================

Compares multiple ML models:

1. Logistic Regression
2. Linear SVM
3. Multinomial Naive Bayes

Goal:
Find the best performing model for app classification.

=====================================================
"""

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report


# =====================================================
# LOAD DATASET
# =====================================================

DATASET_PATH = "app_classifier/dataset/app_titles.csv"

df = pd.read_csv(DATASET_PATH, encoding="latin1")

df = df.dropna(subset=["label"])
df = df[df["label"].str.strip() != ""]

print("\nTraining samples:", len(df))


# =====================================================
# TEXT FEATURE
# =====================================================

df["text"] = df["process_name"].astype(str) + " " + df["window_title"].astype(str)

X = df["text"]
y = df["label"]


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================================
# TF-IDF VECTOR
# =====================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# =====================================================
# MODELS
# =====================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Naive Bayes": MultinomialNB()
}


results = {}


# =====================================================
# TRAIN + EVALUATE
# =====================================================

print("\n================ MODEL COMPARISON ================\n")

for name, model in models.items():

    print(f"Training: {name}")

    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)

    results[name] = acc

    print(f"Accuracy: {acc * 100:.2f}%\n")


# =====================================================
# FINAL RANKING
# =====================================================

print("\n================ FINAL RESULTS ================\n")

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for i, (name, acc) in enumerate(sorted_results, 1):

    print(f"{i}. {name} → {acc * 100:.2f}%")


best_model = sorted_results[0]

print("\nBEST MODEL:")
print(best_model[0], "→", f"{best_model[1] * 100:.2f}%")