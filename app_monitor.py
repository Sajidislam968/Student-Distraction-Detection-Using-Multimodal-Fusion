import time
import psutil
import win32gui
import win32process


class AppMonitor:

    def __init__(self):
        self.focused_apps = [
            "code.exe",
            "devenv.exe",
            "pycharm64.exe",
            "winword.exe",
            "excel.exe",
            "powerpnt.exe",
            "acrord32.exe",
            "notepad.exe"
        ]

        self.distracting_apps = [
            "spotify.exe",
            "vlc.exe",
            "wmplayer.exe"
        ]

        self.distracting_sites = [
            "youtube",
            "facebook",
            "instagram",
            "tiktok",
            "netflix",
            "discord"
        ]

    def get_active_window(self):

        hwnd = win32gui.GetForegroundWindow()

        title = win32gui.GetWindowText(hwnd)

        _, pid = win32process.GetWindowThreadProcessId(hwnd)

        process_name = "Unknown"

        try:
            process_name = psutil.Process(pid).name().lower()

        except Exception:
            pass

        return process_name, title

    def classify(self, process_name, title):

        title_lower = title.lower()

        # Check websites first
        for site in self.distracting_sites:
            if site in title_lower:
                return "DISTRACTED", f"{site.title()} detected"

        # Check applications
        if process_name in self.focused_apps:
            return "FOCUSED", "Study/Productivity application"

        if process_name in self.distracting_apps:
            return "DISTRACTED", "Entertainment application"

        if process_name in [
            "chrome.exe",
            "msedge.exe",
            "firefox.exe",
            "brave.exe"
        ]:
            return "NEUTRAL", "Browser (unknown website)"

        return "NEUTRAL", "Unknown application"

    def start(self):

        print("=" * 60)
        print("Active Application Monitor Started")
        print("Press Ctrl+C to Stop")
        print("=" * 60)

        while True:

            process_name, title = self.get_active_window()

            status, reason = self.classify(process_name, title)

            print("\n" + "-" * 60)
            print(f"Application : {process_name}")
            print(f"Window Title: {title}")
            print(f"Status      : {status}")
            print(f"Reason      : {reason}")

            time.sleep(5)


if __name__ == "__main__":

    monitor = AppMonitor()

    try:
        monitor.start()

    except KeyboardInterrupt:
        print("\nApplication Monitor Stopped.")