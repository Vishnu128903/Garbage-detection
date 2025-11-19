from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="Dataset/data.yaml",
    epochs=50,
    imgsz=640
)
