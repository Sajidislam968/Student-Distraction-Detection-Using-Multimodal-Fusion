import os
import time
from datetime import datetime

import pandas as pd
import psutil
import win32gui
import win32process


# =====================================================
# Configuration
# =====================================================

DATASET_FOLDER = "app_classifier/dataset"
CSV_FILE = os.path.join(DATASET_FOLDER, "app_titles.csv")

os.makedirs(DATASET_FOLDER, exist_ok=True)


# =====================================================
# Ignore Rules (Window Titles)
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
# Ignore Rules (Processes)
# =====================================================

IGNORE_PROCESSES = {
    "shellexperiencehost.exe",
    "searchhost.exe",
    "startmenuexperiencehost.exe",
    "lockapp.exe",
    "applicationframehost.exe"
}


# =====================================================
# Create CSV if it doesn't exist
# =====================================================

if not os.path.exists(CSV_FILE):

    df = pd.DataFrame(columns=[
        "timestamp",
        "process_name",
        "window_title",
        "label"
    ])

    df.to_csv(CSV_FILE, index=False, encoding="utf-8")


# =====================================================
# Get Active Window
# =====================================================

def get_active_window():

    hwnd = win32gui.GetForegroundWindow()
    title = win32gui.GetWindowText(hwnd).strip()

    _, pid = win32process.GetWindowThreadProcessId(hwnd)

    try:
        process_name = psutil.Process(pid).name().lower()
    except Exception:
        process_name = "unknown"

    return process_name, title


# =====================================================
# Ignore Checker (FINAL VERSION)
# =====================================================

def should_ignore(title, process_name):

    title = title.strip()

    if title == "":
        return True

    if len(title) < 3:
        return True

    if title in IGNORE_TITLES:
        return True

    if process_name.lower() in IGNORE_PROCESSES:
        return True

    return False


# =====================================================
# Check Duplicate (FIXED ENCODING)
# =====================================================

def already_exists(title):

    df = pd.read_csv(CSV_FILE, encoding="latin1")  # FIXED

    if df.empty:
        return False

    return title in df["window_title"].values


# =====================================================
# Save Sample (FIXED ENCODING CONSISTENCY)
# =====================================================

def save_sample(process_name, title):

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "process_name": process_name,
        "window_title": title,
        "label": ""
    }

    df = pd.DataFrame([row])

    df.to_csv(
        CSV_FILE,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8"   # FIXED
    )


# =====================================================
# Main Loop
# =====================================================

print("=" * 60)
print(" Application Dataset Collector V2.0 ")
print("=" * 60)
print("Collecting CLEAN + UNIQUE window titles only...")
print("Press Ctrl + C to stop.\n")

last_title = ""

try:

    while True:

        process_name, title = get_active_window()

        if should_ignore(title, process_name):
            time.sleep(2)
            continue

        if title != last_title:

            if already_exists(title):

                print(f"[SKIPPED] {title}")

            else:

                save_sample(process_name, title)
                print(f"[SAVED] {title}")

            last_title = title

        time.sleep(2)

except KeyboardInterrupt:

    print("\nDataset collection finished.")
    print(f"Saved to: {CSV_FILE}")