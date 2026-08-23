"""
=====================================================
Active Window Detection Module
=====================================================

Responsible ONLY for detecting the current active
application and filtering invalid windows.

=====================================================
"""

import psutil
import win32gui
import win32process

from config import (
    IGNORE_KEYWORDS,
    IGNORE_PROCESSES,
    IGNORE_TITLES,
)


# =====================================================
# Active Window
# =====================================================

def get_active_window():
    """
    Returns:

        process_name
        window_title

    Example

        chrome.exe

        YouTube - Google Chrome
    """

    hwnd = win32gui.GetForegroundWindow()

    title = win32gui.GetWindowText(hwnd).strip()

    _, pid = win32process.GetWindowThreadProcessId(hwnd)

    try:
        process_name = psutil.Process(pid).name().lower()

    except Exception:
        process_name = "unknown"

    return process_name, title


# =====================================================
# Ignore Filter
# =====================================================

def should_ignore(process_name, title):
    """
    Returns True if the window should not be classified.
    """

    title = title.strip()

    if title == "":
        return True

    if len(title) < 3:
        return True

    if title in IGNORE_TITLES:
        return True

    if process_name.lower() in IGNORE_PROCESSES:
        return True

    title_lower = title.lower()

    for keyword in IGNORE_KEYWORDS:

        if keyword in title_lower:
            return True

    return False


# =====================================================
# Public Function
# =====================================================

def get_current_application():
    """
    Returns

        process_name
        title

    or

        None

    if ignored.
    """

    process_name, title = get_active_window()

    if should_ignore(process_name, title):
        return None

    return process_name, title