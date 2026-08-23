"""
=====================================================
Application Classifier Configuration
=====================================================
"""

import os

# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

MODEL_PATH = os.path.join(MODEL_DIR, "app_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")

DATASET_PATH = os.path.join(DATASET_DIR, "app_titles.csv")

# =====================================================
# Prediction Settings
# =====================================================

# Predictions below this confidence become "Unknown"
CONFIDENCE_THRESHOLD = 60.0

# Polling interval (seconds)
UPDATE_INTERVAL = 1.0

# Number of important words to display
TOP_FEATURES = 5

# =====================================================
# Ignore Window Titles
# =====================================================

IGNORE_TITLES = {

    "",

    "Start",

    "Search",

    "Task Switching",

    "Task View",

    "Widgets",

    "Notification Center",

    "Program Manager"

}

# =====================================================
# Ignore Loading Screens
# =====================================================

IGNORE_KEYWORDS = {

    "opening",

    "loading",

    "starting",

    "welcome"

}

# =====================================================
# Ignore Processes
# =====================================================

IGNORE_PROCESSES = {

    "shellexperiencehost.exe",

    "searchhost.exe",

    "startmenuexperiencehost.exe",

    "lockapp.exe",

    "applicationframehost.exe"

}

# =====================================================
# Labels
# =====================================================

LABEL_FOCUSED = "Focused"
LABEL_DISTRACTED = "Distracted"
LABEL_NEUTRAL = "Neutral"
LABEL_UNKNOWN = "Unknown"