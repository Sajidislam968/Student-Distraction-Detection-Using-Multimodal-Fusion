"""
=====================================================
Test Custom YOLO Focus Objects Model
=====================================================

Detects:
- book
- cup
- bottle

Run:
    python test_yolo_focus_objects.py
=====================================================
"""

import cv2
from ultralytics import YOLO


MODEL_PATH = "runs/detect/focus_objects_yolov8n/weights/best.pt"

USEFUL_OBJECTS = {
    "book",
    "cup",
    "bottle"
}


def main():
    print("=" * 70)
    print("TESTING CUSTOM YOLO MODEL")
    print("=" * 70)

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Webcam not opened.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(
            frame,
            conf=0.35,
            verbose=False
        )

        detected_items = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = str(model.names[cls_id]).lower()

                confidence = float(box.conf[0])

                if class_name not in USEFUL_OBJECTS:
                    continue

                detected_items.append(class_name)

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                label = f"{class_name} {confidence:.2f}"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        if len(detected_items) == 0:
            status = "NO OBJECT"
        else:
            status = ", ".join(sorted(set(detected_items)))

        cv2.putText(
            frame,
            f"Detected: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Custom YOLO Focus Objects",
            frame
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()