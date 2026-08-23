import cv2
import mediapipe as mp

# ---------------------------
# MediaPipe Setup
# ---------------------------

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

# ---------------------------
# Distraction Settings
# ---------------------------

FPS = 30
DISTRACTION_TIME = 3

DISTRACTION_THRESHOLD = FPS * DISTRACTION_TIME

distraction_frames = 0

# ---------------------------
# Landmark IDs
# ---------------------------

NOSE = 1
LEFT_EYE = 33
RIGHT_EYE = 263

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                nose = face_landmarks.landmark[NOSE]
                left_eye = face_landmarks.landmark[LEFT_EYE]
                right_eye = face_landmarks.landmark[RIGHT_EYE]

                nose_x = int(nose.x * w)
                nose_y = int(nose.y * h)

                left_x = int(left_eye.x * w)
                left_y = int(left_eye.y * h)

                right_x = int(right_eye.x * w)
                right_y = int(right_eye.y * h)

                # ---------------------------
                # Draw Landmarks
                # ---------------------------

                cv2.circle(frame, (nose_x, nose_y), 5, (0, 0, 255), -1)
                cv2.circle(frame, (left_x, left_y), 5, (0, 255, 0), -1)
                cv2.circle(frame, (right_x, right_y), 5, (0, 255, 0), -1)

                cv2.line(
                    frame,
                    (left_x, left_y),
                    (right_x, right_y),
                    (255, 0, 0),
                    2
                )

                # ---------------------------
                # Eye Center
                # ---------------------------

                eye_center_x = (left_x + right_x) // 2

                offset = nose_x - eye_center_x

                # ---------------------------
                # Normalize Offset
                # ---------------------------

                eye_distance = abs(right_x - left_x)

                normalized_offset = (
                    offset / eye_distance
                )

                status = "FOCUSED"
                distracted = False

                # Adjust thresholds if needed

                if normalized_offset > 0.10:
                    status = "LOOKING RIGHT"
                    distracted = True

                elif normalized_offset < -0.10:
                    status = "LOOKING LEFT"
                    distracted = True

                # ---------------------------
                # Distraction Timer
                # ---------------------------

                if distracted:
                    distraction_frames += 1
                else:
                    distraction_frames = 0

                away_time = distraction_frames / FPS

                # ---------------------------
                # Display
                # ---------------------------

                cv2.putText(
                    frame,
                    f"Offset: {normalized_offset:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    status,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Away Time: {away_time:.1f}s",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                if distraction_frames > DISTRACTION_THRESHOLD:

                    cv2.putText(
                        frame,
                        "DISTRACTION ALERT!",
                        (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

        cv2.imshow(
            "Study Distraction Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()