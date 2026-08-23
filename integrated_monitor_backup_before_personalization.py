"""
=====================================================
Integrated Parallel Monitoring System
=====================================================

Runs all modules together and updates one shared_state.

Modules:
1. Vision: face, eyes, drowsiness, head pose
2. YOLO: optional, disabled until custom model is ready
3. Keyboard activity
4. Mouse activity
5. App classifier
6. Audio monitor
7. Automatic labeling
8. Fusion dataset CSV collection

Run:
    python integrated_monitor.py

Controls:
    1 = manual label Focused
    2 = manual label Distracted
    3 = manual label Neutral
    4 = manual label Absent
    0 = clear manual label and return to automatic labeling
    q = quit

Camera window:
    ESC = quit
=====================================================
"""

import os
import sys
import csv
import json
import cv2
import time
import math
import msvcrt
import threading
from datetime import datetime

import mediapipe as mp
from pynput import keyboard, mouse

from shared_state import SharedState


# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

APP_CLASSIFIER_DIR = os.path.join(BASE_DIR, "app_classifier")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
FUSION_ENGINE_DIR = os.path.join(BASE_DIR, "fusion_engine")

if APP_CLASSIFIER_DIR not in sys.path:
    sys.path.insert(0, APP_CLASSIFIER_DIR)

if AUDIO_DIR not in sys.path:
    sys.path.insert(0, AUDIO_DIR)

if FUSION_ENGINE_DIR not in sys.path:
    sys.path.insert(0, FUSION_ENGINE_DIR)


# =====================================================
# Main Settings
# =====================================================

SHOW_CAMERA_WINDOW = True

ENABLE_AUDIO = True
AUDIO_DEVICE_INDEX = 7

ENABLE_YOLO = False
YOLO_MODEL_PATH = None

# Later, after YOLO training:
# ENABLE_YOLO = True
# YOLO_MODEL_PATH = "runs/detect/train/weights/best.pt"

FUSION_DATASET_DIR = os.path.join(BASE_DIR, "fusion_dataset")
FUSION_DATASET_PATH = os.path.join(FUSION_DATASET_DIR, "fusion_live_dataset.csv")

CSV_LOG_INTERVAL = 2.0
DISPLAY_INTERVAL = 2.0

# =====================================================
# Session History Settings
# =====================================================

SESSION_HISTORY_DIR = os.path.join(BASE_DIR, "session_history")
SESSION_HISTORY_CSV = os.path.join(SESSION_HISTORY_DIR, "study_session_history.csv")

# Live dashboard status file.
# The Streamlit dashboard reads this file for real-time final decisions.
RUNTIME_DIR = os.path.join(BASE_DIR, "runtime")
LIVE_STATUS_PATH = os.path.join(RUNTIME_DIR, "live_status.json")


# =====================================================
# Automatic Labeling Settings
# =====================================================

ENABLE_AUTO_LABELING = True

EYES_CLOSED_SLEEP_SECONDS = 30.0
HEAD_AWAY_DISTRACTION_SECONDS = 30.0
NO_FACE_ABSENT_SECONDS = 5.0

LOW_AUDIO_STUDY_PROBABILITY = 40.0
HIGH_AUDIO_STUDY_PROBABILITY = 70.0

# =====================================================
# Hybrid Fusion Decision Settings
# =====================================================
# XGBoost is trusted directly when confidence is high.
# Below this threshold, strong auto-label evidence can correct the decision.
HYBRID_XGBOOST_TRUST_THRESHOLD = 80.0

# For auto Focused correction, app classifier must be confident.
HYBRID_FOCUSED_APP_CONFIDENCE = 70.0

# Long idle prevents weak focused correction.
HYBRID_LONG_IDLE_SECONDS = 60.0


# =====================================================
# Alert Settings
# =====================================================

ENABLE_ALERTS = True
ENABLE_AUDIO_ALERT = True
ENABLE_VISUAL_ALERT = True

# Alert if the final smoothed decision remains Distracted for this long.
DISTRACTION_ALERT_SECONDS = 10.0

# Prevents repeated beeps/notifications every second.
ALERT_COOLDOWN_SECONDS = 20.0


# =====================================================
# Vision Worker
# =====================================================

class VisionWorker(threading.Thread):
    def __init__(self, shared_state, stop_event):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.stop_event = stop_event

        self.mp_face_mesh = mp.solutions.face_mesh

        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        self.NOSE = 1
        self.LEFT_EYE_CENTER = 33
        self.RIGHT_EYE_CENTER = 263

        self.EAR_THRESHOLD = 0.20
        self.DROWSY_FRAMES = 60

        self.FPS = 30
        self.DISTRACTION_TIME = 3
        self.DISTRACTION_THRESHOLD = self.FPS * self.DISTRACTION_TIME

        self.blink_count = 0
        self.eye_closed = False
        self.closed_frames = 0
        self.distraction_frames = 0

        self.yolo_model = None
        self.yolo_names = {}

    def distance(self, p1, p2):
        return math.hypot(
            p1[0] - p2[0],
            p1[1] - p2[1]
        )

    def calculate_ear(self, landmarks, eye_points):
        p1 = landmarks[eye_points[0]]
        p2 = landmarks[eye_points[1]]
        p3 = landmarks[eye_points[2]]
        p4 = landmarks[eye_points[3]]
        p5 = landmarks[eye_points[4]]
        p6 = landmarks[eye_points[5]]

        vertical1 = self.distance(p2, p6)
        vertical2 = self.distance(p3, p5)
        horizontal = self.distance(p1, p4)

        if horizontal == 0:
            return 0

        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear

    def load_yolo_if_enabled(self):
        if not ENABLE_YOLO:
            self.shared_state.update(
                "yolo",
                yolo_status="Disabled",
                phone_detected=0,
                object_detected=0,
                detected_objects=""
            )
            return

        if YOLO_MODEL_PATH is None or not os.path.exists(YOLO_MODEL_PATH):
            self.shared_state.update(
                "yolo",
                yolo_status="Model Missing",
                phone_detected=0,
                object_detected=0,
                detected_objects=""
            )
            return

        try:
            from ultralytics import YOLO

            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            self.yolo_names = self.yolo_model.names

            self.shared_state.update(
                "yolo",
                yolo_status="Loaded"
            )

        except Exception as e:
            self.yolo_model = None

            self.shared_state.update(
                "yolo",
                yolo_status=f"YOLO Error: {e}",
                phone_detected=0,
                object_detected=0,
                detected_objects=""
            )

    def run_yolo(self, frame):
        if self.yolo_model is None:
            return frame, 0, 0, ""

        phone_detected = 0
        object_detected = 0
        detected_objects = []

        try:
            results = self.yolo_model(frame, verbose=False)

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = str(self.yolo_names[cls_id]).lower()

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    useful_objects = {
                        "cell phone",
                        "phone",
                        "mobile phone",
                        "book",
                        "bottle",
                        "cup",
                        "packet"
                    }

                    if class_name in useful_objects:
                        object_detected = 1
                        detected_objects.append(class_name)

                        if class_name in {
                            "cell phone",
                            "phone",
                            "mobile phone"
                        }:
                            phone_detected = 1

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            class_name,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

        except Exception:
            pass

        detected_objects_text = ",".join(sorted(set(detected_objects)))

        return frame, phone_detected, object_detected, detected_objects_text

    def draw_overlay(self, frame, snapshot):
        y = 30

        lines = [
            f"Face: {snapshot.get('face_detected')}",
            f"Eyes: {snapshot.get('eye_status')}",
            f"Head: {snapshot.get('head_status')}",
            f"Drowsy: {snapshot.get('drowsy')}",
            f"Phone: {snapshot.get('phone_detected')}",
            f"App: {snapshot.get('app_label')}",
            f"Audio: {snapshot.get('audio_label')} ({snapshot.get('study_probability')}%)",
            f"Final Decision: {snapshot.get('final_decision', 'Initializing')}",
            f"Alert: {snapshot.get('alert_message', 'No alert')}"
        ]

        for line in lines:
            cv2.putText(
                frame,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            y += 28

        return frame

    def run(self):
        self.load_yolo_if_enabled()

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            self.shared_state.update(
                "vision",
                face_detected=0,
                head_status="Camera Error",
                head_away=1,
                eye_status="Unknown",
                drowsy=0
            )
            return

        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            while not self.stop_event.is_set():
                success, frame = cap.read()

                if not success:
                    time.sleep(0.1)
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                face_detected = 0
                head_status = "No Face"
                head_away = 1
                eye_status = "Unknown"
                drowsy = 0
                away_time = 0.0

                if results.multi_face_landmarks:
                    face_detected = 1
                    head_status = "Forward"
                    head_away = 0

                    face_landmarks = results.multi_face_landmarks[0]

                    landmarks = []
                    for lm in face_landmarks.landmark:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        landmarks.append((x, y))

                    left_ear = self.calculate_ear(
                        landmarks,
                        self.LEFT_EYE
                    )

                    right_ear = self.calculate_ear(
                        landmarks,
                        self.RIGHT_EYE
                    )

                    ear = (left_ear + right_ear) / 2

                    if ear < self.EAR_THRESHOLD:
                        eye_status = "Closed"
                        self.closed_frames += 1

                        if not self.eye_closed:
                            self.eye_closed = True
                    else:
                        eye_status = "Open"

                        if self.eye_closed:
                            self.blink_count += 1

                        self.eye_closed = False
                        self.closed_frames = 0

                    if self.closed_frames > self.DROWSY_FRAMES:
                        drowsy = 1

                    nose_x = landmarks[self.NOSE][0]
                    left_x = landmarks[self.LEFT_EYE_CENTER][0]
                    right_x = landmarks[self.RIGHT_EYE_CENTER][0]

                    eye_center_x = (left_x + right_x) // 2
                    offset = nose_x - eye_center_x
                    eye_distance = abs(right_x - left_x)

                    if eye_distance > 0:
                        normalized_offset = offset / eye_distance
                    else:
                        normalized_offset = 0

                    distracted = False

                    if normalized_offset > 0.15:
                        head_status = "Looking Right"
                        head_away = 1
                        distracted = True

                    elif normalized_offset < -0.15:
                        head_status = "Looking Left"
                        head_away = 1
                        distracted = True

                    else:
                        head_status = "Forward"
                        head_away = 0

                    if distracted:
                        self.distraction_frames += 1
                    else:
                        self.distraction_frames = 0

                    away_time = self.distraction_frames / self.FPS

                else:
                    self.distraction_frames = 0
                    self.closed_frames = 0
                    self.eye_closed = False

                frame, phone_detected, object_detected, detected_objects = self.run_yolo(frame)

                self.shared_state.update(
                    "vision",
                    face_detected=face_detected,
                    head_status=head_status,
                    head_away=head_away,
                    eye_status=eye_status,
                    drowsy=drowsy,
                    blink_count=self.blink_count,
                    away_time=round(away_time, 2),
                    phone_detected=phone_detected,
                    object_detected=object_detected,
                    detected_objects=detected_objects
                )

                if SHOW_CAMERA_WINDOW:
                    snapshot = self.shared_state.get()
                    frame = self.draw_overlay(frame, snapshot)

                    cv2.imshow(
                        "Integrated Focus Monitor",
                        frame
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key == 27:
                        self.stop_event.set()
                        break

        cap.release()
        cv2.destroyAllWindows()


# =====================================================
# Keyboard Worker
# =====================================================

class KeyboardWorker(threading.Thread):
    def __init__(self, shared_state, stop_event):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.stop_event = stop_event

        self.key_count = 0
        self.previous_count = 0
        self.lock = threading.Lock()

    def on_press(self, key):
        # Global stop shortcut.
        # Press F9 from anywhere to stop the full monitoring system.
        if key == keyboard.Key.f9:
            self.stop_event.set()
            return

        with self.lock:
            self.key_count += 1

    def run(self):
        listener = keyboard.Listener(
            on_press=self.on_press
        )

        listener.start()

        while not self.stop_event.is_set():
            time.sleep(5)

            with self.lock:
                current_count = self.key_count

            keys_last_5_sec = current_count - self.previous_count
            self.previous_count = current_count

            keyboard_active = 1 if keys_last_5_sec > 0 else 0
            keyboard_kpm = keys_last_5_sec * 12

            self.shared_state.update(
                "keyboard",
                keyboard_active=keyboard_active,
                keys_last_5_sec=keys_last_5_sec,
                keyboard_kpm=keyboard_kpm
            )

        listener.stop()


# =====================================================
# Mouse Worker
# =====================================================

class MouseWorker(threading.Thread):
    def __init__(self, shared_state, stop_event):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.stop_event = stop_event

        self.move_count = 0
        self.click_count = 0

        self.previous_move_count = 0
        self.previous_click_count = 0

        self.last_position = None
        self.last_activity_time = time.time()

        self.lock = threading.Lock()

    def on_move(self, x, y):
        with self.lock:
            if self.last_position is None:
                self.last_position = (x, y)
                return

            distance = math.hypot(
                x - self.last_position[0],
                y - self.last_position[1]
            )

            if distance >= 5:
                self.move_count += 1
                self.last_activity_time = time.time()
                self.last_position = (x, y)

    def on_click(self, x, y, button, pressed):
        if pressed:
            with self.lock:
                self.click_count += 1
                self.last_activity_time = time.time()

    def run(self):
        listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click
        )

        listener.start()

        while not self.stop_event.is_set():
            time.sleep(5)

            with self.lock:
                current_moves = self.move_count
                current_clicks = self.click_count
                idle_time = time.time() - self.last_activity_time

            moves_last_5_sec = current_moves - self.previous_move_count
            clicks_last_5_sec = current_clicks - self.previous_click_count

            self.previous_move_count = current_moves
            self.previous_click_count = current_clicks

            mouse_active = 1 if (
                moves_last_5_sec > 0
                or clicks_last_5_sec > 0
            ) else 0

            self.shared_state.update(
                "mouse",
                mouse_active=mouse_active,
                mouse_moves_last_5_sec=moves_last_5_sec,
                mouse_clicks_last_5_sec=clicks_last_5_sec,
                mouse_idle_time=round(idle_time, 2)
            )

        listener.stop()


# =====================================================
# App Classifier Worker
# =====================================================

class AppClassifierWorker(threading.Thread):
    def __init__(self, shared_state, stop_event):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.stop_event = stop_event

        self.last_window = None
        self.last_result = None

    def run(self):
        try:
            from active_window import get_current_application
            from predictor import predict

        except Exception as e:
            self.shared_state.update(
                "app",
                app_label="Unknown",
                app_confidence=0.0,
                app_process="Import Error",
                app_title=str(e)
            )
            return

        while not self.stop_event.is_set():
            try:
                current = get_current_application()

                if current is None:
                    time.sleep(1)
                    continue

                process_name, window_title = current

                title_lower = window_title.lower()

                # Ignore this system's own monitor / terminal window.
                # Do NOT let the system classify itself as Focused.
                if (
                    "integrated focus monitor" in title_lower
                    or "integrated_monitor.py" in title_lower
                    or "integrated parallel focus monitor" in title_lower
                    or "python integrated_monitor.py" in title_lower
                ):
                    self.shared_state.update(
                        "app",
                        app_label="Unknown",
                        app_confidence=0.0,
                        app_process=process_name,
                        app_title=window_title
                    )

                    time.sleep(1)
                    continue

                current_window = (process_name, window_title)

                if current_window != self.last_window:
                    self.last_result = predict(
                        process_name,
                        window_title
                    )

                    self.last_window = current_window

                result = self.last_result

                self.shared_state.update(
                    "app",
                    app_label=result.get("prediction", "Unknown"),
                    app_confidence=round(result.get("confidence", 0.0), 2),
                    app_process=process_name,
                    app_title=window_title
                )

            except Exception as e:
                self.shared_state.update(
                    "app",
                    app_label="Unknown",
                    app_confidence=0.0,
                    app_process="Error",
                    app_title=str(e)
                )

            time.sleep(1)


# =====================================================
# Audio Worker
# =====================================================

class AudioWorker(threading.Thread):
    def __init__(self, shared_state, stop_event):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.stop_event = stop_event

        self.monitor = None

    def run(self):
        if not ENABLE_AUDIO:
            self.shared_state.update(
                "audio",
                speech_status=0,
                audio_label="Audio Disabled",
                study_probability=50.0,
                audio_transcript="",
                audio_method="None"
            )
            return

        try:
            from audio_monitor import AudioMonitor

            self.monitor = AudioMonitor(
                device_index=AUDIO_DEVICE_INDEX
            )

            audio_thread = threading.Thread(
                target=self.monitor.start,
                daemon=True
            )

            audio_thread.start()

            while not self.stop_event.is_set():
                try:
                    data = self.monitor.get_audio_data()

                    self.shared_state.update(
                        "audio",
                        speech_status=1 if data.get("speech_status") else 0,
                        audio_label=data.get(
                            "audio_label",
                            "Uncertain Audio"
                        ),
                        study_probability=float(
                            data.get("study_probability", 50.0)
                        ),
                        audio_transcript=data.get("transcript", ""),
                        audio_method=data.get("method", "None")
                    )

                except Exception as e:
                    self.shared_state.update(
                        "audio",
                        speech_status=0,
                        audio_label="Audio Read Error",
                        study_probability=50.0,
                        audio_transcript=str(e),
                        audio_method="Error"
                    )

                time.sleep(2)

        except Exception as e:
            self.shared_state.update(
                "audio",
                speech_status=0,
                audio_label="Audio Import Error",
                study_probability=50.0,
                audio_transcript=str(e),
                audio_method="Error"
            )

        finally:
            try:
                if self.monitor is not None:
                    self.monitor.running = False
            except Exception:
                pass


# =====================================================
# Automatic Labeling Worker
# =====================================================

class AutoLabelWorker(threading.Thread):
    def __init__(self, shared_state, stop_event):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.stop_event = stop_event

        self.eyes_closed_start = None
        self.head_away_start = None
        self.no_face_start = None

    def update_timer(self, condition, start_time):
        current_time = time.time()

        if condition:
            if start_time is None:
                start_time = current_time

            elapsed = current_time - start_time
        else:
            start_time = None
            elapsed = 0.0

        return start_time, elapsed

    def decide_label(self, snapshot):
        face_detected = int(snapshot.get("face_detected", 0))
        head_away = int(snapshot.get("head_away", 0))
        drowsy = int(snapshot.get("drowsy", 0))
        phone_detected = int(snapshot.get("phone_detected", 0))

        eye_status = str(snapshot.get("eye_status", "Unknown"))
        app_label = str(snapshot.get("app_label", "Unknown"))
        app_confidence = float(snapshot.get("app_confidence", 0.0))
        mouse_idle_time = float(snapshot.get("mouse_idle_time", 0.0))

        keyboard_active = int(snapshot.get("keyboard_active", 0))
        mouse_active = int(snapshot.get("mouse_active", 0))
        speech_status = int(snapshot.get("speech_status", 0))

        study_probability = float(
            snapshot.get("study_probability", 50.0)
        )

        eyes_closed_time = float(
            snapshot.get("eyes_closed_time", 0.0)
        )

        head_away_time = float(
            snapshot.get("head_away_time", 0.0)
        )

        no_face_time = float(
            snapshot.get("no_face_time", 0.0)
        )

        # =================================================
        # Priority 1: Absent
        # =================================================
        if face_detected == 0 and no_face_time >= NO_FACE_ABSENT_SECONDS:
            return (
                "Absent",
                f"No face detected for {no_face_time:.1f} seconds"
            )

        # =================================================
        # Priority 2: Strong distracted signals
        # =================================================
        if phone_detected == 1:
            return (
                "Distracted",
                "Phone detected"
            )

        if drowsy == 1:
            return (
                "Distracted",
                "Drowsiness detected"
            )

        if (
            eye_status == "Closed"
            and eyes_closed_time >= EYES_CLOSED_SLEEP_SECONDS
        ):
            return (
                "Distracted",
                f"Eyes closed for {eyes_closed_time:.1f} seconds"
            )

        if (
            head_away == 1
            and head_away_time >= HEAD_AWAY_DISTRACTION_SECONDS
        ):
            return (
                "Distracted",
                f"Head away for {head_away_time:.1f} seconds"
            )

        if app_label == "Distracted":
            return (
                "Distracted",
                "Distracting application or website detected"
            )

        if (
            speech_status == 1
            and study_probability < LOW_AUDIO_STUDY_PROBABILITY
        ):
            return (
                "Distracted",
                "Non-study audio detected"
            )

        # =================================================
        # Priority 3: Focused
        # =================================================
        good_face_condition = (
            face_detected == 1
            and head_away == 0
            and drowsy == 0
            and phone_detected == 0
            and eye_status != "Closed"
        )

        focused_app_condition = (
            app_label == "Focused"
            and app_confidence >= 70.0
        )

        focused_audio_condition = (
            study_probability >= HIGH_AUDIO_STUDY_PROBABILITY
        )

        behavior_active = (
            keyboard_active == 1
            or mouse_active == 1
        )

        long_idle = mouse_idle_time >= 60.0

        if (
            good_face_condition
            and focused_app_condition
            and behavior_active
        ):
            return (
                "Focused",
                "Focused app with keyboard or mouse activity"
            )

        if (
            good_face_condition
            and focused_app_condition
            and not long_idle
        ):
            return (
                "Focused",
                "Face forward with confident focused application"
            )

        if good_face_condition and focused_audio_condition:
            return (
                "Focused",
                "Face forward with study-related audio"
            )

        # =================================================
        # Priority 4: Neutral
        # =================================================
        return (
            "Neutral",
            "No strong focused or distracted signal"
        )

    def run(self):
        while not self.stop_event.is_set():
            if not ENABLE_AUTO_LABELING:
                time.sleep(0.5)
                continue

            snapshot = self.shared_state.get()

            face_detected = int(snapshot.get("face_detected", 0))
            head_away = int(snapshot.get("head_away", 0))
            eye_status = str(snapshot.get("eye_status", "Unknown"))

            self.no_face_start, no_face_time = self.update_timer(
                face_detected == 0,
                self.no_face_start
            )

            self.head_away_start, head_away_time = self.update_timer(
                head_away == 1,
                self.head_away_start
            )

            self.eyes_closed_start, eyes_closed_time = self.update_timer(
                eye_status == "Closed",
                self.eyes_closed_start
            )

            self.shared_state.update(
                "autolabel_timer",
                no_face_time=round(no_face_time, 2),
                head_away_time=round(head_away_time, 2),
                eyes_closed_time=round(eyes_closed_time, 2)
            )

            snapshot = self.shared_state.get()

            auto_label, auto_reason = self.decide_label(snapshot)

            manual_label = str(
                snapshot.get("manual_label", "")
            ).strip()

            if manual_label != "":
                final_label = manual_label
                final_label_source = "manual"
            else:
                final_label = auto_label
                final_label_source = "auto"

            self.shared_state.update(
                "autolabel",
                auto_label=auto_label,
                auto_reason=auto_reason,
                final_label=final_label,
                final_label_source=final_label_source
            )

            time.sleep(0.5)

# =====================================================
# XGBoost Fusion Decision Worker
# =====================================================

class FusionDecisionWorker(threading.Thread):
    def __init__(self, shared_state, stop_event):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.stop_event = stop_event

        self.predictor = None

        # Smoothing
        self.last_raw_prediction = None
        self.same_prediction_count = 0
        self.stable_decision = "Neutral"

        # Time logging
        self.last_time_update = time.time()

        self.time_spent = {
            "Focused": 0.0,
            "Distracted": 0.0,
            "Neutral": 0.0,
            "Absent": 0.0,
        }

    def load_predictor(self):
        try:
            from fusion_predictor import FusionPredictor

            self.predictor = FusionPredictor()

            self.shared_state.update(
                "fusion",
                fusion_prediction="Model Loaded",
                fusion_confidence=0.0,
                final_decision="Neutral",
                decision_stability_count=0,
                decision_source="Hybrid Fusion: XGBoost + strong auto evidence + smoothing"
            )

            return True

        except Exception as e:
            self.shared_state.update(
                "fusion",
                fusion_prediction="Fusion Model Error",
                fusion_confidence=0.0,
                fusion_probabilities={},
                final_decision="Neutral",
                decision_stability_count=0,
                decision_source=f"Error: {e}"
            )

            return False

    def update_smoothing(self, candidate_prediction):
        """
        Only changes final stable decision if the same candidate
        appears 3 times continuously.
        """

        if candidate_prediction == self.last_raw_prediction:
            self.same_prediction_count += 1
        else:
            self.last_raw_prediction = candidate_prediction
            self.same_prediction_count = 1

        if self.same_prediction_count >= 3:
            self.stable_decision = candidate_prediction

        return self.stable_decision, self.same_prediction_count

    def update_time_logging(self):
        """
        Adds elapsed time to the current stable decision.
        """

        current_time = time.time()
        elapsed = current_time - self.last_time_update
        self.last_time_update = current_time

        if self.stable_decision in self.time_spent:
            self.time_spent[self.stable_decision] += elapsed

        self.shared_state.update(
            "fusion_time",
            focused_time=round(self.time_spent["Focused"], 2),
            distracted_time=round(self.time_spent["Distracted"], 2),
            neutral_time=round(self.time_spent["Neutral"], 2),
            absent_time=round(self.time_spent["Absent"], 2)
        )

    def get_probability(self, probabilities, label):
        """
        Safely reads probability percentage for one class.
        """
        try:
            return float(probabilities.get(label, 0.0))
        except Exception:
            return 0.0

    def has_strong_focused_evidence(self, snapshot):
        """
        Strong Focused evidence:
        - face detected
        - head forward
        - eyes open
        - not drowsy
        - no phone
        - confident focused app
        - some recent keyboard/mouse activity OR not long idle
        """

        face_detected = int(snapshot.get("face_detected", 0))
        head_away = int(snapshot.get("head_away", 0))
        drowsy = int(snapshot.get("drowsy", 0))
        phone_detected = int(snapshot.get("phone_detected", 0))

        eye_status = str(snapshot.get("eye_status", "Unknown"))
        app_label = str(snapshot.get("app_label", "Unknown"))
        app_confidence = float(snapshot.get("app_confidence", 0.0))

        keyboard_active = int(snapshot.get("keyboard_active", 0))
        mouse_active = int(snapshot.get("mouse_active", 0))
        mouse_idle_time = float(snapshot.get("mouse_idle_time", 999.0))

        good_vision = (
            face_detected == 1
            and head_away == 0
            and drowsy == 0
            and phone_detected == 0
            and eye_status != "Closed"
        )

        confident_focused_app = (
            app_label == "Focused"
            and app_confidence >= HYBRID_FOCUSED_APP_CONFIDENCE
        )

        behavior_active = (
            keyboard_active == 1
            or mouse_active == 1
        )

        not_long_idle = mouse_idle_time < HYBRID_LONG_IDLE_SECONDS

        return (
            good_vision
            and confident_focused_app
            and (
                behavior_active
                or not_long_idle
            )
        )

    def has_strong_distracted_evidence(self, snapshot):
        """
        Strong Distracted evidence:
        - phone detected
        - drowsiness
        - distracting app
        - long head-away
        - long eyes-closed
        - non-study speech/audio
        """

        phone_detected = int(snapshot.get("phone_detected", 0))
        drowsy = int(snapshot.get("drowsy", 0))
        head_away = int(snapshot.get("head_away", 0))

        app_label = str(snapshot.get("app_label", "Unknown"))
        eye_status = str(snapshot.get("eye_status", "Unknown"))

        eyes_closed_time = float(snapshot.get("eyes_closed_time", 0.0))
        head_away_time = float(snapshot.get("head_away_time", 0.0))

        speech_status = int(snapshot.get("speech_status", 0))
        study_probability = float(snapshot.get("study_probability", 50.0))

        if phone_detected == 1:
            return True

        if drowsy == 1:
            return True

        if app_label == "Distracted":
            return True

        if (
            eye_status == "Closed"
            and eyes_closed_time >= EYES_CLOSED_SLEEP_SECONDS
        ):
            return True

        if (
            head_away == 1
            and head_away_time >= HEAD_AWAY_DISTRACTION_SECONDS
        ):
            return True

        if (
            speech_status == 1
            and study_probability < LOW_AUDIO_STUDY_PROBABILITY
        ):
            return True

        return False

    def has_strong_absent_evidence(self, snapshot):
        """
        Strong Absent evidence:
        - no face detected for configured absent duration.
        """

        face_detected = int(snapshot.get("face_detected", 0))
        no_face_time = float(snapshot.get("no_face_time", 0.0))

        return (
            face_detected == 0
            and no_face_time >= NO_FACE_ABSENT_SECONDS
        )

    def hybrid_decision(self, raw_prediction, confidence, probabilities, snapshot):
        """
        Final candidate decision before smoothing.

        Logic:
        1. Trust XGBoost directly when confidence is high.
        2. When XGBoost is not highly confident, allow only STRONG
           auto-evidence to correct it.
        3. Never blindly copy auto_label.
        """

        auto_label = str(snapshot.get("auto_label", "Neutral"))

        focused_prob = self.get_probability(probabilities, "Focused")
        distracted_prob = self.get_probability(probabilities, "Distracted")
        absent_prob = self.get_probability(probabilities, "Absent")
        neutral_prob = self.get_probability(probabilities, "Neutral")

        xgboost_is_high_confidence = (
            confidence >= HYBRID_XGBOOST_TRUST_THRESHOLD
        )

        # -------------------------------------------------
        # High-confidence XGBoost: trust model
        # -------------------------------------------------
        if xgboost_is_high_confidence:
            return (
                raw_prediction,
                (
                    f"Hybrid Fusion: trusted high-confidence XGBoost "
                    f"({confidence:.2f}%)"
                )
            )

        # -------------------------------------------------
        # Strong Absent correction
        # -------------------------------------------------
        if (
            auto_label == "Absent"
            and self.has_strong_absent_evidence(snapshot)
        ):
            return (
                "Absent",
                (
                    f"Hybrid Fusion: strong absent evidence corrected "
                    f"XGBoost {raw_prediction} ({confidence:.2f}%)"
                )
            )

        # -------------------------------------------------
        # Strong Distracted correction
        # -------------------------------------------------
        if (
            auto_label == "Distracted"
            and self.has_strong_distracted_evidence(snapshot)
            and raw_prediction != "Distracted"
        ):
            return (
                "Distracted",
                (
                    f"Hybrid Fusion: strong distracted evidence corrected "
                    f"XGBoost {raw_prediction} ({confidence:.2f}%)"
                )
            )

        # -------------------------------------------------
        # Strong Focused correction
        # -------------------------------------------------
        # This fixes cases like:
        # PowerPoint/PDF/VS Code + face forward + active mouse/keyboard
        # but XGBoost weakly says Distracted.
        if (
            auto_label == "Focused"
            and self.has_strong_focused_evidence(snapshot)
            and raw_prediction != "Focused"
        ):
            # Do not override if XGBoost is extremely close to certain.
            # This block only runs when confidence < 80 by design.
            return (
                "Focused",
                (
                    f"Hybrid Fusion: strong focused evidence corrected "
                    f"XGBoost {raw_prediction} ({confidence:.2f}%, "
                    f"Focused {focused_prob:.2f}%, Distracted {distracted_prob:.2f}%)"
                )
            )

        # -------------------------------------------------
        # Otherwise use XGBoost
        # -------------------------------------------------
        return (
            raw_prediction,
            (
                f"Hybrid Fusion: XGBoost used "
                f"({confidence:.2f}%; F={focused_prob:.2f}, "
                f"D={distracted_prob:.2f}, N={neutral_prob:.2f}, A={absent_prob:.2f})"
            )
        )

    def run(self):
        model_loaded = self.load_predictor()

        if not model_loaded:
            return

        while not self.stop_event.is_set():
            try:
                snapshot = self.shared_state.get()

                result = self.predictor.predict(snapshot)

                raw_prediction = result["fusion_prediction"]
                confidence = result["fusion_confidence"]
                probabilities = result["fusion_probabilities"]

                candidate_decision, decision_source = self.hybrid_decision(
                    raw_prediction,
                    confidence,
                    probabilities,
                    snapshot
                )

                final_decision, stability_count = self.update_smoothing(
                    candidate_decision
                )

                self.update_time_logging()

                self.shared_state.update(
                    "fusion",
                    fusion_prediction=raw_prediction,
                    fusion_confidence=confidence,
                    fusion_probabilities=probabilities,
                    final_decision=final_decision,
                    decision_stability_count=stability_count,
                    decision_source=decision_source
                )

            except Exception as e:
                self.shared_state.update(
                    "fusion",
                    fusion_prediction="Prediction Error",
                    fusion_confidence=0.0,
                    fusion_probabilities={},
                    final_decision=self.stable_decision,
                    decision_source=f"Prediction Error: {e}"
                )

            time.sleep(1)



# =====================================================
# Alert Worker
# =====================================================

class AlertWorker(threading.Thread):
    def __init__(self, shared_state, stop_event):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.stop_event = stop_event

        self.distracted_start = None
        self.last_alert_time = 0.0
        self.alert_count = 0

    def play_audio_alert(self):
        """
        Windows audio alert using built-in winsound.
        Runs in a separate short thread so monitoring does not freeze.
        """
        if not ENABLE_AUDIO_ALERT:
            return

        try:
            import winsound

            winsound.Beep(1000, 250)
            time.sleep(0.10)
            winsound.Beep(1200, 250)
            time.sleep(0.10)
            winsound.Beep(1000, 250)

        except Exception:
            try:
                import winsound
                winsound.MessageBeep()
            except Exception:
                pass

    def run(self):
        while not self.stop_event.is_set():
            if not ENABLE_ALERTS:
                time.sleep(0.5)
                continue

            snapshot = self.shared_state.get()
            final_decision = str(snapshot.get("final_decision", "Neutral"))
            current_time = time.time()

            if final_decision == "Distracted":
                if self.distracted_start is None:
                    self.distracted_start = current_time

                distracted_duration = current_time - self.distracted_start

                should_alert = (
                    distracted_duration >= DISTRACTION_ALERT_SECONDS
                    and current_time - self.last_alert_time >= ALERT_COOLDOWN_SECONDS
                )

                if should_alert:
                    self.last_alert_time = current_time
                    self.alert_count += 1

                    alert_message = (
                        f"Distraction alert: distracted for "
                        f"{distracted_duration:.1f} seconds"
                    )

                    self.shared_state.update(
                        "alert",
                        alert_active=1,
                        alert_message=alert_message,
                        alert_count=self.alert_count,
                        distracted_alert_duration=round(distracted_duration, 2),
                        last_alert_time=current_time
                    )

                    threading.Thread(
                        target=self.play_audio_alert,
                        daemon=True
                    ).start()

                else:
                    self.shared_state.update(
                        "alert",
                        alert_active=0,
                        alert_message=(
                            f"Distracted for {distracted_duration:.1f} sec; "
                            f"alert threshold {DISTRACTION_ALERT_SECONDS:.0f} sec"
                        ),
                        alert_count=self.alert_count,
                        distracted_alert_duration=round(distracted_duration, 2)
                    )

            else:
                self.distracted_start = None

                self.shared_state.update(
                    "alert",
                    alert_active=0,
                    alert_message="No alert",
                    alert_count=self.alert_count,
                    distracted_alert_duration=0.0
                )

            time.sleep(0.5)


# =====================================================
# Live Status Writer
# =====================================================

class LiveStatusWriter(threading.Thread):
    def __init__(self, shared_state, stop_event):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.stop_event = stop_event

        os.makedirs(RUNTIME_DIR, exist_ok=True)

    def run(self):
        while not self.stop_event.is_set():
            try:
                snapshot = self.shared_state.get()
                snapshot["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                temp_path = LIVE_STATUS_PATH + ".tmp"

                with open(
                    temp_path,
                    "w",
                    encoding="utf-8"
                ) as f:
                    json.dump(
                        snapshot,
                        f,
                        indent=2,
                        default=str
                    )

                os.replace(temp_path, LIVE_STATUS_PATH)

            except Exception:
                pass

            time.sleep(1)


# =====================================================
# Fusion CSV Logger
# =====================================================

class FusionCSVLogger(threading.Thread):
    def __init__(self, shared_state, stop_event):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.stop_event = stop_event

        os.makedirs(FUSION_DATASET_DIR, exist_ok=True)

        self.header = [
            "timestamp",

            "face_detected",
            "head_away",
            "eyes_closed_time",
            "head_away_time",
            "no_face_time",
            "drowsy",
            "phone_detected",

            "keyboard_active",
            "mouse_active",

            "app_focused",
            "app_distracted",
            "app_neutral",
            "app_unknown",
            "app_confidence",

            "speech_status",
            "audio_study_probability",

            "head_status",
            "eye_status",
            "app_label",
            "audio_label",
            "detected_objects",

            "auto_label",
            "auto_reason",
            "final_label_source",
            "final_label"
        ]

        self.ensure_dataset_file()

    def ensure_dataset_file(self):
        """
        Creates the CSV if missing.

        If an old CSV exists with a different header, it is backed up
        so the new automatic-label rows do not mix with old columns.
        """
        if not os.path.exists(FUSION_DATASET_PATH):
            self.write_header()
            return

        if os.path.getsize(FUSION_DATASET_PATH) == 0:
            self.write_header()
            return

        try:
            with open(
                FUSION_DATASET_PATH,
                "r",
                newline="",
                encoding="utf-8"
            ) as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)

            if existing_header != self.header:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = FUSION_DATASET_PATH.replace(
                    ".csv",
                    f"_backup_{timestamp}.csv"
                )

                os.rename(
                    FUSION_DATASET_PATH,
                    backup_path
                )

                self.write_header()

                print(
                    "\nOld fusion CSV header was different."
                )
                print(
                    f"Old file backed up to: {backup_path}"
                )
                print(
                    f"New CSV created at: {FUSION_DATASET_PATH}\n"
                )

        except Exception:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = FUSION_DATASET_PATH.replace(
                ".csv",
                f"_backup_{timestamp}.csv"
            )

            try:
                os.rename(
                    FUSION_DATASET_PATH,
                    backup_path
                )
            except Exception:
                pass

            self.write_header()

    def write_header(self):
        with open(
            FUSION_DATASET_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def build_row(self, snapshot):
        app_label = str(
            snapshot.get("app_label", "Unknown")
        )

        app_focused = 1 if app_label == "Focused" else 0
        app_distracted = 1 if app_label == "Distracted" else 0
        app_neutral = 1 if app_label == "Neutral" else 0
        app_unknown = 1 if app_label == "Unknown" else 0

        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            snapshot.get("face_detected", 0),
            snapshot.get("head_away", 0),
            snapshot.get("eyes_closed_time", 0.0),
            snapshot.get("head_away_time", 0.0),
            snapshot.get("no_face_time", 0.0),
            snapshot.get("drowsy", 0),
            snapshot.get("phone_detected", 0),

            snapshot.get("keyboard_active", 0),
            snapshot.get("mouse_active", 0),

            app_focused,
            app_distracted,
            app_neutral,
            app_unknown,
            snapshot.get("app_confidence", 0.0),

            snapshot.get("speech_status", 0),
            snapshot.get("study_probability", 50.0),

            snapshot.get("head_status", "Unknown"),
            snapshot.get("eye_status", "Unknown"),
            snapshot.get("app_label", "Unknown"),
            snapshot.get("audio_label", "Uncertain Audio"),
            snapshot.get("detected_objects", ""),

            snapshot.get("auto_label", "Neutral"),
            snapshot.get("auto_reason", ""),
            snapshot.get("final_label_source", "auto"),
            snapshot.get("final_label", "")
        ]

        return row

    def run(self):
        while not self.stop_event.is_set():
            snapshot = self.shared_state.get()

            final_label = str(
                snapshot.get("final_label", "")
            ).strip()

            if final_label != "":
                row = self.build_row(snapshot)

                with open(
                    FUSION_DATASET_PATH,
                    "a",
                    newline="",
                    encoding="utf-8"
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

            time.sleep(CSV_LOG_INTERVAL)


# =====================================================
# Console Display
# =====================================================

def clear_console():
    os.system("cls")


def format_seconds(seconds):
    seconds = int(float(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return f"{hours:02}:{minutes:02}:{seconds:02}"


def print_dashboard(snapshot):
    clear_console()

    print("=" * 80)
    print("              INTEGRATED PARALLEL FOCUS MONITOR")
    print("=" * 80)

    print("\nCONTROLS")
    print("-" * 80)
    print("1=Focused | 2=Distracted | 3=Neutral | 4=Absent | 0=Auto Mode | q=Quit | F9=Stop Anywhere")

    print("\nCURRENT FINAL LABEL FOR FUSION DATASET")
    print("-" * 80)
    print(f"Final Label : {snapshot.get('final_label', '')}")
    print(f"Source      : {snapshot.get('final_label_source', '')}")
    print(f"Auto Label  : {snapshot.get('auto_label', '')}")
    print(f"Reason      : {snapshot.get('auto_reason', '')}")
    print(f"Manual      : {snapshot.get('manual_label', '')}")

    print("\nXGBOOST FINAL DECISION")
    print("-" * 80)
    print(f"Raw Prediction   : {snapshot.get('fusion_prediction')}")
    print(f"Confidence       : {snapshot.get('fusion_confidence')}%")
    print(f"Final Decision   : {snapshot.get('final_decision')}")
    print(f"Stable Count     : {snapshot.get('decision_stability_count')}")
    print(f"Decision Source  : {snapshot.get('decision_source')}")

    probabilities = snapshot.get("fusion_probabilities", {})
    if probabilities:
        print("\nFusion Probabilities")
        for label, probability in probabilities.items():
            print(f"{label:<12}: {probability:>6.2f}%")

    print("\nSESSION TIME LOGGING")
    print("-" * 80)
    print(f"Focused Time    : {format_seconds(snapshot.get('focused_time', 0.0))}")
    print(f"Distracted Time : {format_seconds(snapshot.get('distracted_time', 0.0))}")
    print(f"Neutral Time    : {format_seconds(snapshot.get('neutral_time', 0.0))}")
    print(f"Absent Time     : {format_seconds(snapshot.get('absent_time', 0.0))}")

    print("\nVISION")
    print("-" * 80)
    print(f"Face Detected    : {snapshot.get('face_detected')}")
    print(f"Head Status      : {snapshot.get('head_status')}")
    print(f"Head Away        : {snapshot.get('head_away')}")
    print(f"Eye Status       : {snapshot.get('eye_status')}")
    print(f"Drowsy           : {snapshot.get('drowsy')}")
    print(f"Blink Count      : {snapshot.get('blink_count')}")
    print(f"Away Time        : {snapshot.get('away_time')} sec")
    print(f"Eyes Closed Time : {snapshot.get('eyes_closed_time')} sec")
    print(f"Head Away Time   : {snapshot.get('head_away_time')} sec")
    print(f"No Face Time     : {snapshot.get('no_face_time')} sec")

    print("\nYOLO / OBJECT")
    print("-" * 80)
    print(f"YOLO Status      : {snapshot.get('yolo_status')}")
    print(f"Phone Detected   : {snapshot.get('phone_detected')}")
    print(f"Object Detected  : {snapshot.get('object_detected')}")
    print(f"Detected Objects : {snapshot.get('detected_objects')}")

    print("\nBEHAVIOR")
    print("-" * 80)
    print(f"Keyboard Active : {snapshot.get('keyboard_active')}")
    print(f"Keys Last 5 sec : {snapshot.get('keys_last_5_sec')}")
    print(f"Keyboard KPM    : {snapshot.get('keyboard_kpm')}")
    print(f"Mouse Active    : {snapshot.get('mouse_active')}")
    print(f"Mouse Moves     : {snapshot.get('mouse_moves_last_5_sec')}")
    print(f"Mouse Clicks    : {snapshot.get('mouse_clicks_last_5_sec')}")
    print(f"Mouse Idle Time : {snapshot.get('mouse_idle_time')} sec")

    print("\nAPPLICATION")
    print("-" * 80)
    print(f"App Label      : {snapshot.get('app_label')}")
    print(f"App Confidence : {snapshot.get('app_confidence')}%")
    print(f"Process        : {snapshot.get('app_process')}")
    print(f"Window         : {snapshot.get('app_title')}")

    print("\nAUDIO")
    print("-" * 80)
    print(f"Speech Status     : {snapshot.get('speech_status')}")
    print(f"Audio Label       : {snapshot.get('audio_label')}")
    print(f"Study Probability : {snapshot.get('study_probability')}%")
    print(f"Method            : {snapshot.get('audio_method')}")
    print(f"Transcript        : {snapshot.get('audio_transcript')}")

    print("\nCSV")
    print("-" * 80)
    print(f"Fusion dataset saved to: {FUSION_DATASET_PATH}")

    print("=" * 80)


def handle_keyboard_input(shared_state, stop_event):
    if not msvcrt.kbhit():
        return

    key = msvcrt.getch().decode(errors="ignore").lower()

    if key == "1":
        shared_state.update(
            "manual",
            manual_label="Focused"
        )

    elif key == "2":
        shared_state.update(
            "manual",
            manual_label="Distracted"
        )

    elif key == "3":
        shared_state.update(
            "manual",
            manual_label="Neutral"
        )

    elif key == "4":
        shared_state.update(
            "manual",
            manual_label="Absent"
        )

    elif key == "0":
        shared_state.update(
            "manual",
            manual_label=""
        )

    elif key == "q":
        stop_event.set()


# =====================================================
# Session History Saving
# =====================================================

def save_session_history(shared_state):
    os.makedirs(SESSION_HISTORY_DIR, exist_ok=True)

    snapshot = shared_state.get()

    session_start_time = snapshot.get("session_start_time", time.time())
    session_end_time = time.time()
    duration_seconds = session_end_time - session_start_time

    header = [
        "session_end_time",
        "duration_seconds",
        "duration_hhmmss",
        "focused_time",
        "distracted_time",
        "neutral_time",
        "absent_time",
        "final_decision",
        "last_fusion_prediction",
        "last_fusion_confidence",
        "alert_count"
    ]

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        round(duration_seconds, 2),
        format_seconds(duration_seconds),
        round(snapshot.get("focused_time", 0.0), 2),
        round(snapshot.get("distracted_time", 0.0), 2),
        round(snapshot.get("neutral_time", 0.0), 2),
        round(snapshot.get("absent_time", 0.0), 2),
        snapshot.get("final_decision", "Unknown"),
        snapshot.get("fusion_prediction", "Unknown"),
        snapshot.get("fusion_confidence", 0.0),
        snapshot.get("alert_count", 0),
    ]

    file_exists = os.path.exists(SESSION_HISTORY_CSV)

    with open(
        SESSION_HISTORY_CSV,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(header)

        writer.writerow(row)

    summary_path = os.path.join(
        SESSION_HISTORY_DIR,
        f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write("Study Session Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Session End Time : {row[0]}\n")
        f.write(f"Duration         : {row[2]}\n\n")

        f.write("Time Spent\n")
        f.write("-" * 50 + "\n")
        f.write(f"Focused    : {format_seconds(snapshot.get('focused_time', 0.0))}\n")
        f.write(f"Distracted : {format_seconds(snapshot.get('distracted_time', 0.0))}\n")
        f.write(f"Neutral    : {format_seconds(snapshot.get('neutral_time', 0.0))}\n")
        f.write(f"Absent     : {format_seconds(snapshot.get('absent_time', 0.0))}\n\n")

        f.write("Final Decision\n")
        f.write("-" * 50 + "\n")
        f.write(f"Final Decision   : {snapshot.get('final_decision', 'Unknown')}\n")
        f.write(f"Fusion Prediction: {snapshot.get('fusion_prediction', 'Unknown')}\n")
        f.write(f"Confidence       : {snapshot.get('fusion_confidence', 0.0)}%\n")

    print(f"Session history saved to: {SESSION_HISTORY_CSV}")
    print(f"Session summary saved to: {summary_path}")


# =====================================================
# Main
# =====================================================

def main():
    shared_state = SharedState()
    stop_event = threading.Event()

    workers = [
        VisionWorker(shared_state, stop_event),
        KeyboardWorker(shared_state, stop_event),
        MouseWorker(shared_state, stop_event),
        AppClassifierWorker(shared_state, stop_event),
        AudioWorker(shared_state, stop_event),
        AutoLabelWorker(shared_state, stop_event),
        FusionDecisionWorker(shared_state, stop_event),
        AlertWorker(shared_state, stop_event),
        LiveStatusWriter(shared_state, stop_event),
        FusionCSVLogger(shared_state, stop_event),
    ]

    print("=" * 80)
    print("Starting Integrated Parallel Monitoring...")
    print("=" * 80)

    for worker in workers:
        worker.start()

    try:
        last_display_time = 0

        while not stop_event.is_set():
            handle_keyboard_input(
                shared_state,
                stop_event
            )

            current_time = time.time()

            if current_time - last_display_time >= DISPLAY_INTERVAL:
                snapshot = shared_state.get()
                print_dashboard(snapshot)
                last_display_time = current_time

            time.sleep(0.1)

    except KeyboardInterrupt:
        stop_event.set()

    finally:
        stop_event.set()
        time.sleep(1)

        save_session_history(shared_state)

        print("\nIntegrated Monitoring Stopped.")
        print(f"Fusion dataset saved at: {FUSION_DATASET_PATH}")
        print(f"Session history saved at: {SESSION_HISTORY_CSV}")


if __name__ == "__main__":
    main()
