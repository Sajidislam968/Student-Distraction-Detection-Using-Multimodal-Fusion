"""
=====================================================
Console Display Module
=====================================================

Responsible ONLY for displaying prediction results.

=====================================================
"""


import os
from datetime import datetime

from colorama import Fore, Style, init

from config import (
    LABEL_FOCUSED,
    LABEL_DISTRACTED,
    LABEL_NEUTRAL,
    LABEL_UNKNOWN,
)


os.system("cls")  # Windows
init(autoreset=True)



# =====================================================
# Prediction History
# =====================================================

prediction_history = []



# =====================================================
# Colors
# =====================================================

STATUS = {

    LABEL_FOCUSED: ("🟢", Fore.GREEN),

    LABEL_DISTRACTED: ("🔴", Fore.RED),

    LABEL_NEUTRAL: ("🟡", Fore.YELLOW),

    LABEL_UNKNOWN: ("🟣", Fore.MAGENTA),

}



# =====================================================
# Confidence Bar
# =====================================================

def confidence_bar(confidence, length=20):

    filled = int((confidence / 100) * length)

    bar = "█" * filled + "░" * (length - filled)

    return f"[{bar}] {confidence:.2f}%"




# =====================================================
# Probability Distribution
# =====================================================

def print_probabilities(probabilities):

    print("\nProbability Distribution")
    print("-" * 45)


    sorted_probs = sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    )


    for i, (label, probability) in enumerate(sorted_probs, start=1):

        print(
            f"{i}. {label:<12} {probability:>6.2f}%"
        )




# =====================================================
# AI Explanation
# =====================================================

def print_features(features):

    print("\nTop AI Features")
    print("-" * 45)


    if not features:

        print("No important features found.")

        return


    for i, feature in enumerate(features, start=1):

        print(
            f"{i}. {feature}"
        )




# =====================================================
# Recent Activity History
# =====================================================

def print_history():

    print("\nRECENT ACTIVITY")
    print("-" * 45)


    for item in prediction_history:

        print(
            f"{item['time']}  "
            f"{item['window']:<30} "
            f"{item['prediction']}"
        )




# =====================================================
# Format Seconds
# =====================================================

def format_time(seconds):

    seconds = int(seconds)

    hours = seconds // 3600

    minutes = (seconds % 3600) // 60

    seconds = seconds % 60

    return f"{hours:02}:{minutes:02}:{seconds:02}"




# =====================================================
# Session Summary
# =====================================================

def print_statistics(stats):

    print("\nSESSION SUMMARY")
    print("-" * 45)


    print(
        f"Duration          : {stats['duration']}"
    )


    print(
        f"Total Predictions : {stats['total']}"
    )


    print()


    counts = stats["counts"]

    percentages = stats["percentages"]


    labels = [

        LABEL_FOCUSED,

        LABEL_DISTRACTED,

        LABEL_NEUTRAL,

        LABEL_UNKNOWN,

    ]



    # ---------------------------------
    # Prediction Count Statistics
    # ---------------------------------

    for label in labels:

        count = counts.get(label, 0)

        percentage = percentages.get(label, 0)


        print(

            f"{label:<12}: "

            f"{count:>4} "

            f"({percentage:>5.1f}%)"

        )



    # ---------------------------------
    # Time Spent Statistics
    # ---------------------------------

    if "time_spent" in stats:

        print()

        print("TIME SPENT")

        print("-" * 45)


        time_spent = stats["time_spent"]


        for label in labels:

            seconds = time_spent.get(label, 0)


            print(

                f"{label:<12}: "

                f"{format_time(seconds)}"

            )




# =====================================================
# Main Display
# =====================================================

def display_prediction(

    process_name,

    window_title,

    result,

    statistics

):


    global prediction_history


    prediction = result["prediction"]



    # -------------------------------
    # Save current prediction history
    # -------------------------------

    prediction_history.append(

        {

            "time": datetime.now().strftime("%H:%M:%S"),

            "window": window_title[:30],

            "prediction": prediction

        }

    )



    # Keep only latest 5 activities

    if len(prediction_history) > 5:

        prediction_history.pop(0)



    icon, color = STATUS.get(

        prediction,

        ("❓", Fore.WHITE)

    )



    print("\n")

    print("=" * 70)

    print("              AI APPLICATION CLASSIFIER")

    print("=" * 70)



    print(

        f"Timestamp : "

        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    )


    print("-" * 70)



    print(

        f"Process : {process_name}"

    )


    print()



    print("Window")

    print(window_title)



    print("-" * 70)



    print(

        f"Prediction : "

        f"{color}{icon} {prediction}{Style.RESET_ALL}"

    )



    print(

        f"Confidence : "

        f"{confidence_bar(result['confidence'])}"

    )



    print_probabilities(

        result["probabilities"]

    )



    print_features(

        result["features"]

    )



    # Show previous activities

    print_history()



    print_statistics(

        statistics

    )



    print("=" * 70)