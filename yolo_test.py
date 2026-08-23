from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

# -------------------------
# Focus / distraction score
# -------------------------
focus_score = 100

DISTRACTION_OBJECTS = {
    "cell phone": -30,
    "bottle": -10,
    "wine glass": -10,
    "laptop": +5  # studying signal
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    distracted = False
    detected_items = []

    for r in results:
        for box in r.boxes:

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # -------------------------
            # FILTER ONLY USEFUL OBJECTS
            # -------------------------
            if class_name in DISTRACTION_OBJECTS:

                detected_items.append(class_name)

                # draw box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(
                    frame,
                    class_name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

                # -------------------------
                # UPDATE FOCUS SCORE
                # -------------------------
                focus_score += DISTRACTION_OBJECTS[class_name]

                if class_name == "cell phone":
                    distracted = True

    # -------------------------
    # Clamp score
    # -------------------------
    focus_score = max(0, min(100, focus_score))

    # -------------------------
    # STATUS LOGIC
    # -------------------------
    if "cell phone" in detected_items:
        status = "PHONE DISTRACTION"

    elif "bottle" in detected_items:
        status = "DRINKING"

    elif "wine glass" in detected_items:
        status = "DRINKING (CUP)"

    elif len(detected_items) == 0:
        status = "FOCUSED"

    else:
        status = "UNKNOWN OBJECT DETECTED"

    # -------------------------
    # DISPLAY
    # -------------------------
    cv2.putText(frame, f"Focus Score: {focus_score}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(frame, f"Status: {status}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("YOLO Focus Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()