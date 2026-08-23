from pynput import mouse
import time
import math


class MouseMonitor:
    def __init__(self):
        # Statistics
        self.move_count = 0
        self.click_count = 0

        # Mouse position
        self.last_position = None

        # Time tracking
        self.start_time = time.time()
        self.last_activity_time = time.time()

        # Real-time statistics
        self.previous_move_count = 0
        self.previous_click_count = 0

    # Called whenever the mouse moves
    def on_move(self, x, y):
        if self.last_position is None:
            self.last_position = (x, y)
            return

        distance = math.hypot(
            x - self.last_position[0],
            y - self.last_position[1]
        )

        # Ignore tiny movements (helps reduce noise)
        if distance >= 5:
            self.move_count += 1
            self.last_activity_time = time.time()
            self.last_position = (x, y)

    # Called whenever a mouse button is clicked
    def on_click(self, x, y, button, pressed):
        if pressed:
            self.click_count += 1
            self.last_activity_time = time.time()

    def start(self):
        listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click
        )

        listener.start()

        print("=" * 50)
        print("        Mouse Activity Monitor Started")
        print("        Press Ctrl + C to Stop")
        print("=" * 50)

        while True:
            time.sleep(5)

            elapsed = time.time() - self.start_time

            idle_time = time.time() - self.last_activity_time

            current_moves = self.move_count
            current_clicks = self.click_count

            moves_last_5_sec = (
                current_moves - self.previous_move_count
            )

            clicks_last_5_sec = (
                current_clicks - self.previous_click_count
            )

            self.previous_move_count = current_moves
            self.previous_click_count = current_clicks

            print("\n" + "-" * 50)
            print(f"Elapsed Time         : {elapsed:.1f} sec")
            print(f"Total Mouse Moves    : {self.move_count}")
            print(f"Moves (Last 5 Sec)   : {moves_last_5_sec}")
            print(f"Total Clicks         : {self.click_count}")
            print(f"Clicks (Last 5 Sec)  : {clicks_last_5_sec}")
            print(f"Idle Time            : {idle_time:.1f} sec")

            if idle_time >= 60:
                print("Status               : POSSIBLE DISTRACTION")
            else:
                print("Status               : ACTIVE")

            print("-" * 50)


if __name__ == "__main__":
    monitor = MouseMonitor()

    try:
        monitor.start()

    except KeyboardInterrupt:
        print("\nMouse Monitor Stopped.")