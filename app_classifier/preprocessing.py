"""
=====================================================
Text Preprocessing Module
=====================================================

This module prepares application names and window titles
before they are sent to the AI model.

The SAME preprocessing MUST be used during:
    • Training
    • Prediction

=====================================================
"""

import re
import string


# =====================================================
# Remove Punctuation
# =====================================================

TRANSLATION_TABLE = str.maketrans(
    string.punctuation,
    " " * len(string.punctuation)
)


# =====================================================
# Clean Text
# =====================================================

def clean_text(text: str) -> str:
    """
    Clean text for machine learning.

    Example:

    Input:
        chrome.exe YouTube - Google Chrome

    Output:
        chrome exe youtube google chrome
    """

    if text is None:
        return ""

    text = str(text)

    # Lowercase
    text = text.lower()

    # Replace punctuation with spaces
    text = text.translate(TRANSLATION_TABLE)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =====================================================
# Combine Process + Window Title
# =====================================================

def prepare_text(process_name: str, window_title: str) -> str:
    """
    Build a single ML input string.

    Example:

    code.exe
    FocusMonitor - Visual Studio Code

    becomes

    code exe focusmonitor visual studio code
    """

    combined = f"{process_name} {window_title}"

    return clean_text(combined)