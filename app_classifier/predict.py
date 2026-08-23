"""
=====================================================
AI Application Classifier
=====================================================

Main application.

Responsibilities

1. Detect active application
2. Ignore invalid windows
3. Predict using AI
4. Update statistics
5. Display results

=====================================================
"""

import os
import time

from active_window import get_current_application
from predictor import predict
from display import display_prediction
from statistics import SessionStatistics

from config import UPDATE_INTERVAL


# =====================================================
# Start
# =====================================================

print("=" * 70)
print("          AI APPLICATION CLASSIFIER")
print("=" * 70)

print("\nLoading modules...")

stats = SessionStatistics()

last_window = None

print("Ready.")
print("\nPress Ctrl+C to stop.\n")

try:

    while True:

        current = get_current_application()

        if current is None:

            time.sleep(UPDATE_INTERVAL)

            continue

        process_name, window_title = current

        current_window = (process_name, window_title)

        # Ignore duplicate windows
        if current_window == last_window:

            time.sleep(UPDATE_INTERVAL)

            continue

        # AI Prediction
        result = predict(process_name, window_title)

        # Update statistics
        stats.update(result["prediction"])

        # Clear console
        os.system("cls")

        # Display
        display_prediction(

            process_name,

            window_title,

            result,

            stats.get_statistics()

        )

        last_window = current_window

        time.sleep(UPDATE_INTERVAL)

except KeyboardInterrupt:

    print("\n")

    print("=" * 70)

    print("Application Classifier Stopped")

    print("=" * 70)

    print("\nFinal Session Summary\n")

    summary = stats.get_statistics()

    counts = summary["counts"]

    percentages = summary["percentages"]

    print(f"Duration          : {summary['duration']}")

    print(f"Total Predictions : {summary['total']}")

    print()

    for label in counts:

        print(

            f"{label:<12}"

            f"{counts[label]:>6}"

            f" ({percentages[label]:5.1f}%)"

        )

    print("\nGoodbye!")