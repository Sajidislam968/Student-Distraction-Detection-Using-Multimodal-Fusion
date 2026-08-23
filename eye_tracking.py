import cv2
import mediapipe as mp
import math

# ---------------------------
# MediaPipe Face Mesh Setup
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh

# ---------------------------
# Distance Function
# ---------------------------
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ---------------------------
# Eye Aspect Ratio (EAR)
# ---------------------------
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

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear

# ---------------------------
# Eye Landmark Indices
# ---------------------------

# Left Eye
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# Right Eye
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ---------------------------
# Variables
# ---------------------------

blink_count = 0
eye_closed = False

closed_frames = 0

EAR_THRESHOLD = 0.20
DROWSY_FRAMES = 60

# ---------------------------
# Webcam
# ---------------------------

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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        h, w, _ = frame.shape

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                landmarks = []

                for lm in face_landmarks.landmark:

                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    landmarks.append((x, y))

                # Draw eye landmarks
                for idx in LEFT_EYE + RIGHT_EYE:

                    cv2.circle(
                        frame,
                        landmarks[idx],
                        2,
                        (0, 255, 0),
                        -1
                    )

                # -----------------------
                # EAR Calculation
                # -----------------------

                left_ear = calculate_ear(
                    landmarks,
                    LEFT_EYE
                )

                right_ear = calculate_ear(
                    landmarks,
                    RIGHT_EYE
                )

                ear = (left_ear + right_ear) / 2

                # -----------------------
                # Eye Open / Closed
                # -----------------------

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

                # -----------------------
                # Drowsiness Detection
                # -----------------------

                drowsy = False

                if closed_frames > DROWSY_FRAMES:
                    drowsy = True

                # -----------------------
                # Display Information
                # -----------------------

                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Eyes: {eye_status}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Blinks: {blink_count}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 255),
                    2
                )

                if drowsy:

                    cv2.putText(
                        frame,
                        "DROWSY ALERT!",
                        (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

        cv2.imshow(
            "Focus Monitor - Eye Tracking",
            frame
        )

        key = cv2.waitKey(1)

        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()