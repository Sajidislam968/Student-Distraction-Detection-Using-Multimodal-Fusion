from ultralytics import YOLO

print("Loading model...")
model = YOLO("yolov8n.pt")

print("Starting training...")

results = model.train(
    data="dataset/data.yaml",
    epochs=5,
    imgsz=640,
    workers=0,
    device="cpu"
)

print("Training finished")