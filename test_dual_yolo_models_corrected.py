"""
=====================================================
Test Pretrained Phone YOLO + Custom Object YOLO
=====================================================

Pretrained yolov8n.pt detects:
- cell phone

Custom model detects your classes:
0 = bottle
1 = cup
2 = book

Run from FocusMonitor root:
    python test_dual_yolo_models_corrected.py

Press ESC to stop.
=====================================================
"""

from pathlib import Path
import cv2
from ultralytics import YOLO

PRETRAINED_MODEL_PATH = "yolov8n.pt"
CUSTOM_CANDIDATES = [
    Path("custom_yolo_runs/focus_objects_fast/weights/best.pt"),
    Path("custom_yolo_runs/focus_objects_fast/weights/last.pt"),
    Path("runs/detect/runs/detect/focus_objects_yolov8n/weights/best.pt"),
    Path("runs/detect/runs/detect/focus_objects_yolov8n/weights/last.pt"),
    Path("runs/detect/focus_objects_yolov8n/weights/best.pt"),
    Path("runs/detect/focus_objects_yolov8n/weights/last.pt"),
]

YOLO_IMAGE_SIZE = 416
YOLO_CONFIDENCE = 0.35


def get_class_ids(names, target_names):
    target_names = {name.lower() for name in target_names}
    ids = []

    for class_id, class_name in names.items():
        if str(class_name).lower() in target_names:
            ids.append(int(class_id))

    return ids if ids else None


def find_custom_model():
    for path in CUSTOM_CANDIDATES:
        if path.exists():
            return str(path)
    return None


def draw_box(frame, box, label):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def main():
    print("=" * 70)
    print("TESTING DUAL YOLO MODELS")
    print("=" * 70)

    pretrained_model = YOLO(PRETRAINED_MODEL_PATH)
    phone_ids = get_class_ids(
        pretrained_model.names,
        {"cell phone", "phone", "mobile phone"}
    )

    custom_path = find_custom_model()
    custom_model = None
    custom_ids = None

    if custom_path:
        print("Custom model found:", custom_path)
        custom_model = YOLO(custom_path)
        print("Custom model class names:", custom_model.names)
        custom_ids = get_class_ids(
            custom_model.names,
            {"bottle", "cup", "book", "water bottle"}
        )
    else:
        print("Custom model not found. Testing pretrained phone only.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Webcam not opened.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_items = []

        phone_kwargs = {
            "conf": YOLO_CONFIDENCE,
            "imgsz": YOLO_IMAGE_SIZE,
            "verbose": False,
        }
        if phone_ids:
            phone_kwargs["classes"] = phone_ids

        phone_results = pretrained_model(frame, **phone_kwargs)

        for result in phone_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = str(pretrained_model.names[cls_id]).lower()
                confidence = float(box.conf[0])

                if class_name in {"cell phone", "phone", "mobile phone"}:
                    detected_items.append("cell phone")
                    draw_box(frame, box, f"cell phone {confidence:.2f}")

        if custom_model is not None:
            custom_kwargs = {
                "conf": YOLO_CONFIDENCE,
                "imgsz": YOLO_IMAGE_SIZE,
                "verbose": False,
            }
            if custom_ids:
                custom_kwargs["classes"] = custom_ids

            custom_results = custom_model(frame, **custom_kwargs)

            for result in custom_results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = str(custom_model.names[cls_id]).lower()
                    confidence = float(box.conf[0])

                    if class_name == "water bottle":
                        class_name = "bottle"

                    if class_name in {"bottle", "cup", "book"}:
                        detected_items.append(class_name)
                        draw_box(frame, box, f"{class_name} {confidence:.2f}")

        status = ", ".join(sorted(set(detected_items))) if detected_items else "No target object"

        cv2.putText(
            frame,
            f"Detected: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Dual YOLO Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
