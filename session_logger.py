import csv
import time
from datetime import datetime


class SessionLogger:

    def __init__(self):

        self.session_start_time = time.time()

        self.csv_file = open(
            "study_session_log.csv",
            "w",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            "Timestamp",
            "Blink Count",
            "Focus Score",
            "Focus State",
            "Head Status",
            "Drowsy"
        ])

    def log(
        self,
        blink_count,
        focus_score,
        focus_state,
        head_status,
        drowsy
    ):

        self.writer.writerow([
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            blink_count,
            round(focus_score, 1),
            focus_state,
            head_status,
            "YES" if drowsy else "NO"
        ])

        self.csv_file.flush()

    def create_summary(
        self,
        blink_count,
        total_distractions,
        total_drowsy_events,
        average_focus
    ):

        session_end_time = time.time()

        duration_minutes = (
            session_end_time -
            self.session_start_time
        ) / 60

        if average_focus >= 80:
            performance = "GOOD"
        elif average_focus >= 60:
            performance = "AVERAGE"
        else:
            performance = "POOR"

        with open(
            "study_session_summary.txt",
            "w",
            encoding="utf-8"
        ) as summary:

            summary.write(
                "Study Session Summary\n"
            )

            summary.write(
                "=====================\n\n"
            )

            summary.write(
                f"Duration: "
                f"{duration_minutes:.2f} minutes\n"
            )

            summary.write(
                f"Total Blinks: "
                f"{blink_count}\n"
            )

            summary.write(
                f"Distraction Events: "
                f"{total_distractions}\n"
            )

            summary.write(
                f"Drowsy Events: "
                f"{total_drowsy_events}\n"
            )

            summary.write(
                f"Average Focus Score: "
                f"{average_focus:.1f}%\n"
            )

            summary.write(
                f"Overall Performance: "
                f"{performance}\n"
            )

    def close(self):

        if not self.csv_file.closed:
            self.csv_file.close()