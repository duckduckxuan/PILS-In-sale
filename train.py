from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="garbage.yaml",
    epochs=50,
    imgsz=640,
    project="runs/detect",
    name="best",
    device='cpu'
    # device=0
)
