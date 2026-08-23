"""
=====================================================
Shared State Module
=====================================================
Stores latest outputs from all modules in one place.

This is the central bridge for:
- Vision
- YOLO
- Keyboard
- Mouse
- App classifier
- Audio
- Automatic labeling
- XGBoost fusion prediction
- Session time logging
=====================================================
"""

import time
import threading
from copy import deepcopy


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()

        self.state = {
            # -------------------------
            # Vision
            # -------------------------
            "face_detected": 0,
            "head_status": "Unknown",
            "head_away": 0,
            "eye_status": "Unknown",
            "drowsy": 0,
            "blink_count": 0,
            "away_time": 0.0,

            # -------------------------
            # Vision timers
            # -------------------------
            "eyes_closed_time": 0.0,
            "head_away_time": 0.0,
            "no_face_time": 0.0,

            # -------------------------
            # YOLO / Object Detection
            # -------------------------
            "phone_detected": 0,
            "object_detected": 0,
            "detected_objects": "",
            "yolo_status": "Disabled",

            # -------------------------
            # Keyboard
            # -------------------------
            "keyboard_active": 0,
            "keys_last_5_sec": 0,
            "keyboard_kpm": 0.0,

            # -------------------------
            # Mouse
            # -------------------------
            "mouse_active": 0,
            "mouse_moves_last_5_sec": 0,
            "mouse_clicks_last_5_sec": 0,
            "mouse_idle_time": 0.0,

            # -------------------------
            # Application Classifier
            # -------------------------
            "app_label": "Unknown",
            "app_confidence": 0.0,
            "app_process": "",
            "app_title": "",

            # -------------------------
            # Audio
            # -------------------------
            "speech_status": 0,
            "audio_label": "Uncertain Audio",
            "study_probability": 50.0,
            "audio_transcript": "",
            "audio_method": "None",

            # -------------------------
            # Auto/manual dataset labels
            # -------------------------
            "auto_label": "Neutral",
            "auto_reason": "Initializing",
            "manual_label": "",
            "final_label": "",
            "final_label_source": "auto",

            # -------------------------
            # XGBoost Fusion Decision
            # -------------------------
            "fusion_prediction": "Initializing",
            "fusion_confidence": 0.0,
            "fusion_probabilities": {},
            "final_decision": "Initializing",
            "decision_stability_count": 0,
            "decision_source": "XGBoost + Smoothing",

            # -------------------------
            # Final decision time logging
            # -------------------------
            "focused_time": 0.0,
            "distracted_time": 0.0,
            "neutral_time": 0.0,
            "absent_time": 0.0,

            # -------------------------
            # Metadata
            # -------------------------
            "session_start_time": time.time(),
            "last_updated": time.time(),
        }

    def update(self, source_name, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                self.state[key] = value

            now = time.time()
            self.state["last_updated"] = now
            self.state[f"{source_name}_updated"] = now

    def get(self):
        with self.lock:
            return deepcopy(self.state)

    def set_final_label(self, label):
        """
        Backward-compatible function.
        """
        with self.lock:
            self.state["manual_label"] = label
            self.state["final_label"] = label
            self.state["final_label_source"] = "manual" if label else "auto"
            self.state["last_updated"] = time.time()