import cv2
import mediapipe as mp
import math
import time

from session_logger import SessionLogger

# ==================================
# MediaPipe Setup
# ==================================

mp_face_mesh = mp.solutions.face_mesh

# ==================================
# Distance Function
# ==================================

def distance(p1, p2):
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )

# ==================================
# EAR Calculation
# ==================================

def calculate_ear(landmarks, eye_points):

    p1 = landmarks[eye_points[0]]
    p2 = landmarks[eye_points[1]]
    p3 = landmarks[eye_points[2]]
    p4 = landmarks[eye_points[3]]
    p5 = landmarks[eye_points[4]]
    p6 = landmarks[eye_points[5]]

    vertical1 = distance(p2, p6)
    vertical2 = distance(p3, p5)
    horizontal = distance(p1, p4)

    ear = (vertical1 + vertical2) / (2 * horizontal)

    return ear

# ==================================
# Eye Landmarks
# ==================================

LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# ==================================
# Distraction Landmarks
# ==================================

NOSE = 1
LEFT_EYE_CENTER = 33
RIGHT_EYE_CENTER = 263

# ==================================
# Eye Tracking Variables
# ==================================

blink_count = 0
eye_closed = False

closed_frames = 0

EAR_THRESHOLD = 0.20
DROWSY_FRAMES = 60

# ==================================
# Distraction Variables
# ==================================

FPS = 30

DISTRACTION_TIME = 3
DISTRACTION_THRESHOLD = FPS * DISTRACTION_TIME

distraction_frames = 0

# ==================================
# Session Statistics
# ==================================

total_frames = 0
focus_sum = 0

last_log_time = time.time()

total_distractions = 0
total_drowsy_events = 0

distraction_active = False
drowsy_active = False

# ==================================
# Webcam
# ==================================

logger = SessionLogger()

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        # ============================
        # Default Values
        # ============================

        focus_score = 100

        eye_status = "OPEN"
        head_status = "FOCUSED"

        drowsy = False
        distracted = False

        away_time = 0

        # ============================
        # Face Found
        # ============================

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                landmarks = []

                for lm in face_landmarks.landmark:

                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    landmarks.append((x, y))

                # ==================================
                # Draw Eye Landmarks
                # ==================================

                for idx in LEFT_EYE_EAR + RIGHT_EYE_EAR:

                    cv2.circle(
                        frame,
                        landmarks[idx],
                        2,
                        (0, 255, 0),
                        -1
                    )

                # ==================================
                # EAR
                # ==================================

                left_ear = calculate_ear(
                    landmarks,
                    LEFT_EYE_EAR
                )

                right_ear = calculate_ear(
                    landmarks,
                    RIGHT_EYE_EAR
                )

                ear = (left_ear + right_ear) / 2

                # ==================================
                # Eye Status
                # ==================================

                if ear < EAR_THRESHOLD:

                    eye_status = "CLOSED"

                    closed_frames += 1

                    if not eye_closed:
                        eye_closed = True

                else:

                    eye_status = "OPEN"

                    if eye_closed:
                        blink_count += 1
                        eye_closed = False

                    closed_frames = 0

                # ==================================
                # Drowsiness
                # ==================================

                if closed_frames > DROWSY_FRAMES:

                    drowsy = True

                    if not drowsy_active:
                        total_drowsy_events += 1
                        drowsy_active = True

                else:

                    drowsy_active = False

                # ==================================
                # Head Distraction
                # ==================================

                nose_x = landmarks[NOSE][0]

                left_x = landmarks[LEFT_EYE_CENTER][0]
                right_x = landmarks[RIGHT_EYE_CENTER][0]

                eye_center_x = (
                    left_x + right_x
                ) // 2

                offset = (
                    nose_x - eye_center_x
                )

                eye_distance = abs(
                    right_x - left_x
                )

                if eye_distance > 0:

                    normalized_offset = (
                        offset / eye_distance
                    )

                else:

                    normalized_offset = 0

                if normalized_offset > 0.15:

                    head_status = "LOOKING RIGHT"
                    distracted = True

                elif normalized_offset < -0.15:

                    head_status = "LOOKING LEFT"
                    distracted = True

                # ==================================
                # Distraction Timer
                # ==================================

                if distracted:
                    distraction_frames += 1
                else:
                    distraction_frames = 0

                away_time = (
                    distraction_frames / FPS
                )

                # ==================================
                # Focus Score
                # ==================================

                if eye_status == "CLOSED":
                    focus_score -= 20

                if drowsy:
                    focus_score -= 40

                if distracted:
                    focus_score -= 15

                if distraction_frames > DISTRACTION_THRESHOLD:

                    focus_score -= 25

                    if not distraction_active:
                        total_distractions += 1
                        distraction_active = True

                else:

                    distraction_active = False

                focus_score = max(
                    0,
                    focus_score
                )

                # ==================================
                # Focus State
                # ==================================

                if focus_score >= 80:

                    focus_state = "HIGH FOCUS"

                elif focus_score >= 60:

                    focus_state = "MODERATE FOCUS"

                else:

                    focus_state = "LOW FOCUS"

                # ==================================
                # CSV Logging Every 5 Seconds
                # ==================================

                current_time = time.time()

                if current_time - last_log_time >= 5:

                    logger.log(
                        blink_count,
                        focus_score,
                        focus_state,
                        head_status,
                        drowsy
                    )

                    last_log_time = current_time

                # ==================================
                # Session Statistics
                # ==================================

                total_frames += 1
                focus_sum += focus_score

                average_focus = (
                    focus_sum / total_frames
                )

                # ==================================
                # Display
                # ==================================

                y = 30

                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

                y += 35

                cv2.putText(
                    frame,
                    f"Eyes: {eye_status}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,0),
                    2
                )

                y += 35

                cv2.putText(
                    frame,
                    f"Blinks: {blink_count}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,0,255),
                    2
                )

                y += 35

                cv2.putText(
                    frame,
                    head_status,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2
                )

                y += 35

                cv2.putText(
                    frame,
                    f"Away: {away_time:.1f}s",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2
                )

                y += 35

                cv2.putText(
                    frame,
                    f"Focus Score: {focus_score}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

                y += 35

                cv2.putText(
                    frame,
                    focus_state,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,0),
                    2
                )

                y += 35

                cv2.putText(
                    frame,
                    f"Session Focus: {average_focus:.1f}%",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,255),
                    2
                )

                if drowsy:

                    cv2.putText(
                        frame,
                        "DROWSY ALERT!",
                        (350, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3
                    )

                if distraction_frames > DISTRACTION_THRESHOLD:

                    cv2.putText(
                        frame,
                        "DISTRACTION ALERT!",
                        (300, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3
                    )

        else:

            focus_score = 50

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

        cv2.imshow(
            "Focus Monitoring System",
            frame
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

average_focus = (
    focus_sum / total_frames
) if total_frames > 0 else 0

logger.create_summary(
    blink_count,
    total_distractions,
    total_drowsy_events,
    average_focus
)

logger.close()

cap.release()
cv2.destroyAllWindows()