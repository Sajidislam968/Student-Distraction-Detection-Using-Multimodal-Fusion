"""
=====================================================
FocusMonitor Unified Launcher
=====================================================

Starts both:
1. Streamlit dashboard
2. Integrated multimodal monitoring backend

Run from FocusMonitor root:
    python run_focusmonitor.py

Stop:
    - Press F9 from anywhere, or
    - Press ESC in camera window, or
    - Press Ctrl+C in this launcher terminal.

When the monitor stops, this launcher also stops the dashboard.
=====================================================
"""

import os
import sys
import time
import signal
import webbrowser
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_PATH = BASE_DIR / "dashboard" / "dashboard.py"
MONITOR_PATH = BASE_DIR / "integrated_monitor.py"
DASHBOARD_URL = "http://localhost:8501"


def check_required_files():
    missing = []

    if not DASHBOARD_PATH.exists():
        missing.append(str(DASHBOARD_PATH))

    if not MONITOR_PATH.exists():
        missing.append(str(MONITOR_PATH))

    if missing:
        print("ERROR: Required file(s) missing:")
        for path in missing:
            print(" -", path)
        print("\nMake sure dashboard.py and integrated_monitor.py are in the correct folders.")
        sys.exit(1)


def start_dashboard():
    print("=" * 80)
    print("Starting Streamlit dashboard...")
    print("=" * 80)

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD_PATH),
        "--server.address=localhost",
        "--server.port=8501",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]

    return subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )


def start_monitor():
    print("=" * 80)
    print("Starting integrated monitoring backend...")
    print("=" * 80)

    command = [
        sys.executable,
        str(MONITOR_PATH),
    ]

    return subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )


def stop_process(process, name):
    if process is None:
        return

    if process.poll() is not None:
        return

    print(f"Stopping {name}...")

    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
            time.sleep(1)

            if process.poll() is None:
                process.terminate()
        else:
            process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def main():
    check_required_files()

    dashboard_process = None
    monitor_process = None

    try:
        dashboard_process = start_dashboard()

        # Give Streamlit a few seconds to start.
        time.sleep(4)
        webbrowser.open(DASHBOARD_URL)

        print("Dashboard URL:", DASHBOARD_URL)
        print("\nNow starting monitor. Use F9 or ESC to stop.\n")

        monitor_process = start_monitor()

        # Keep launcher open until monitor exits.
        monitor_process.wait()

    except KeyboardInterrupt:
        print("\nLauncher interrupted by user.")

    finally:
        stop_process(monitor_process, "integrated monitor")
        stop_process(dashboard_process, "dashboard")

        print("\nFocusMonitor unified system stopped.")


if __name__ == "__main__":
    main()
