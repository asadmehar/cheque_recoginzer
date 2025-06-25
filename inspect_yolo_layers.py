from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # Load your YOLOv8/YOLO11 model
model = model.model.model   # Access the underlying nn.Sequential

for i, m in enumerate(model):
    print(f"Layer {i}: {m.__class__.__name__}")
