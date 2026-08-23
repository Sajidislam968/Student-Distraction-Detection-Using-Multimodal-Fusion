"""
=====================================================
YOLO Custom Training for FocusMonitor Objects
=====================================================

Classes:
0 = book
1 = cup
2 = bottle

Run:
    python train_yolo_focus_objects.py
=====================================================
"""

from ultralytics import YOLO


def main():
    print("=" * 70)
    print("YOLO CUSTOM OBJECT TRAINING")
    print("=" * 70)

    print("Loading pretrained YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    print("Starting training...")

    results = model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        workers=0,
        device="cpu",
        project="runs/detect",
        name="focus_objects_yolov8n",
        exist_ok=True
    )

    print("=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print("Best model should be saved at:")
    print("runs/detect/focus_objects_yolov8n/weights/best.pt")


if __name__ == "__main__":
    main()