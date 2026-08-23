"""
=====================================================
Fast YOLO Custom Training for FocusMonitor Objects
=====================================================

Your class order:
0 = bottle
1 = cup
2 = book

Run from FocusMonitor root:
    python train_yolo_focus_objects_fast_corrected.py

Output:
    custom_yolo_runs/focus_objects_fast/weights/best.pt
=====================================================
"""

from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path("dataset_fast/data.yaml")


def main():
    print("=" * 70)
    print("FAST YOLO CUSTOM OBJECT TRAINING")
    print("=" * 70)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            "dataset_fast/data.yaml not found. Run this first:\n"
            "    python make_yolo_small_dataset_corrected.py"
        )

    print("Loading pretrained YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    print("Starting fast CPU training...")
    print("Classes: 0=bottle, 1=cup, 2=book")

    results = model.train(
        data=str(DATA_YAML),
        epochs=5,
        imgsz=320,
        batch=4,
        workers=0,
        device="cpu",
        project="custom_yolo_runs",
        name="focus_objects_fast",
        exist_ok=True,
        patience=3,
        cache=False,
        plots=True,
        val=True,
    )

    print("=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print("Best model should be saved at:")
    print("custom_yolo_runs/focus_objects_fast/weights/best.pt")
    print("Last model should be saved at:")
    print("custom_yolo_runs/focus_objects_fast/weights/last.pt")


if __name__ == "__main__":
    main()
