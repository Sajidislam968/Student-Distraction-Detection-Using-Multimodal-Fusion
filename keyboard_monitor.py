from pynput import keyboard
import time


class KeyboardMonitor:
    def __init__(self):
        # Total statistics
        self.key_count = 0
        self.start_time = time.time()

        # Real-time statistics
        self.previous_count = 0

    # Called whenever a key is pressed
    def on_press(self, key):
        self.key_count += 1

    def start(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        print("=" * 45)
        print("      Keyboard Activity Monitor Started")
        print("      Press Ctrl + C to Stop")
        print("=" * 45)

        while True:
            time.sleep(5)

            # Total elapsed time
            elapsed = time.time() - self.start_time

            # Overall typing speed
            overall_kpm = (
                (self.key_count / elapsed) * 60
                if elapsed > 0 else 0
            )

            # Keys typed in the last 5 seconds
            current_count = self.key_count
            keys_last_5_sec = current_count - self.previous_count
            self.previous_count = current_count

            # Real-time typing speed
            current_kpm = keys_last_5_sec * 12

            print("\n" + "-" * 45)
            print(f"Elapsed Time          : {elapsed:.1f} sec")
            print(f"Total Keys Pressed    : {self.key_count}")
            print(f"Keys (Last 5 Seconds) : {keys_last_5_sec}")
            print(f"Current Typing Speed  : {current_kpm} keys/min")
            print(f"Overall Average Speed : {overall_kpm:.2f} keys/min")
            print("-" * 45)


if __name__ == "__main__":
    monitor = KeyboardMonitor()

    try:
        monitor.start()

    except KeyboardInterrupt:
        print("\nKeyboard Monitor Stopped.")