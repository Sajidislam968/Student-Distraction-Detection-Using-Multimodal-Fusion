"""
=====================================================
Session Statistics Module
=====================================================

Tracks prediction statistics during the current session.

=====================================================
"""

import time

from config import (
    LABEL_FOCUSED,
    LABEL_DISTRACTED,
    LABEL_NEUTRAL,
    LABEL_UNKNOWN,
)


class SessionStatistics:

    def __init__(self):

        # Session start
        self.start_time = time.time()

        # Previous prediction
        self.last_prediction = None

        # Time when last prediction started
        self.last_timestamp = time.time()

        # Prediction counts
        self.total_predictions = 0

        self.stats = {

            LABEL_FOCUSED: 0,

            LABEL_DISTRACTED: 0,

            LABEL_NEUTRAL: 0,

            LABEL_UNKNOWN: 0

        }

        # Time spent in each state

        self.time_stats = {

            LABEL_FOCUSED: 0.0,

            LABEL_DISTRACTED: 0.0,

            LABEL_NEUTRAL: 0.0,

            LABEL_UNKNOWN: 0.0

        }


    # =====================================================
    # Update Statistics
    # =====================================================

    def update(self, prediction):

        current_time = time.time()

        if prediction not in self.stats:

            prediction = LABEL_UNKNOWN


        # Add elapsed time to previous state
        if self.last_prediction is not None:

            elapsed = current_time - self.last_timestamp

            self.time_stats[self.last_prediction] += elapsed


        # Update prediction counters

        self.stats[prediction] += 1

        self.total_predictions += 1


        # Store current prediction state

        self.last_prediction = prediction

        self.last_timestamp = current_time



    # =====================================================
    # Session Time
    # =====================================================

    def session_duration(self):

        seconds = int(time.time() - self.start_time)

        hours = seconds // 3600

        minutes = (seconds % 3600) // 60

        seconds = seconds % 60

        return f"{hours:02}:{minutes:02}:{seconds:02}"



    # =====================================================
    # Percentages
    # =====================================================

    def percentages(self):

        if self.total_predictions == 0:

            return {

                label: 0.0

                for label in self.stats

            }


        return {

            label: round(

                count / self.total_predictions * 100,

                2

            )

            for label, count in self.stats.items()

        }



    # =====================================================
    # Get All Information
    # =====================================================

    def get_statistics(self):

        return {

            "counts": self.stats.copy(),

            "percentages": self.percentages(),

            "time_spent": self.time_stats.copy(),

            "total": self.total_predictions,

            "duration": self.session_duration()

        }