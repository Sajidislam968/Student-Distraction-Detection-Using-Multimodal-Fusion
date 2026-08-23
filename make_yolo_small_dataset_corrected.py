"""
=====================================================
Make Small YOLO Dataset for Fast CPU Training
=====================================================

Your class order from dataset/data.yaml:
0 = bottle
1 = cup
2 = book

This script copies a smaller balanced subset from:
    dataset/images/train
    dataset/labels/train
    dataset/images/val
    dataset/labels/val

to:
    dataset_fast/images/train
    dataset_fast/labels/train
    dataset_fast/images/val
    dataset_fast/labels/val

Run from FocusMonitor root:
    python make_yolo_small_dataset_corrected.py
=====================================================
"""

from pathlib import Path
import random
import shutil

SOURCE_DATASET = Path("dataset")
FAST_DATASET = Path("dataset_fast")

# Small and fast for CPU laptop. Increase later if needed.
TRAIN_IMAGES_PER_CLASS = 50
VAL_IMAGES_PER_CLASS = 15
RANDOM_SEED = 42

CLASS_NAMES = {
    0: "bottle",
    1: "cup",
    2: "book",
}

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def read_classes_from_label(label_path: Path):
    """Return class IDs found in one YOLO label file."""
    class_ids = set()

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label_path.read_text(errors="ignore").splitlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        try:
            class_id = int(float(parts[0]))
        except Exception:
            continue

        if class_id in CLASS_NAMES:
            class_ids.add(class_id)

    return class_ids


def find_image_for_label(image_dir: Path, stem: str):
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def collect_pairs_by_class(split: str):
    image_dir = SOURCE_DATASET / "images" / split
    label_dir = SOURCE_DATASET / "labels" / split

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image folder: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label folder: {label_dir}")

    by_class = {class_id: [] for class_id in CLASS_NAMES}

    for label_path in label_dir.glob("*.txt"):
        image_path = find_image_for_label(image_dir, label_path.stem)
        if image_path is None:
            continue

        class_ids = read_classes_from_label(label_path)

        for class_id in class_ids:
            by_class[class_id].append((image_path, label_path))

    return by_class


def copy_split(split: str, images_per_class: int):
    by_class = collect_pairs_by_class(split)

    selected = []
    selected_stems = set()

    for class_id, pairs in by_class.items():
        random.shuffle(pairs)
        target_count = min(images_per_class, len(pairs))

        added_for_class = 0

        for image_path, label_path in pairs:
            if image_path.stem in selected_stems:
                continue

            selected.append((image_path, label_path))
            selected_stems.add(image_path.stem)
            added_for_class += 1

            if added_for_class >= target_count:
                break

        print(
            f"{split} | class {class_id} ({CLASS_NAMES[class_id]}): "
            f"selected {added_for_class} / available {len(pairs)}"
        )

    out_image_dir = FAST_DATASET / "images" / split
    out_label_dir = FAST_DATASET / "labels" / split

    if out_image_dir.exists():
        shutil.rmtree(out_image_dir)
    if out_label_dir.exists():
        shutil.rmtree(out_label_dir)

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    for image_path, label_path in selected:
        shutil.copy2(image_path, out_image_dir / image_path.name)
        shutil.copy2(label_path, out_label_dir / label_path.name)

    print(f"{split}: copied {len(selected)} total images")


def write_yaml():
    yaml_text = f"""path: {FAST_DATASET.resolve().as_posix()}

train: images/train
val: images/val

names:
  0: bottle
  1: cup
  2: book
"""

    yaml_path = FAST_DATASET / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"Created: {yaml_path}")


def main():
    print("=" * 70)
    print("MAKING SMALL FAST YOLO DATASET")
    print("=" * 70)
    print("Class order: 0=bottle, 1=cup, 2=book")

    random.seed(RANDOM_SEED)

    copy_split("train", TRAIN_IMAGES_PER_CLASS)
    copy_split("val", VAL_IMAGES_PER_CLASS)
    write_yaml()

    print("=" * 70)
    print("DONE")
    print("Fast dataset path:", FAST_DATASET.resolve())
    print("Next command:")
    print("    python train_yolo_focus_objects_fast_corrected.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
