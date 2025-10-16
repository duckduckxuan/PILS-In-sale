from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="datasets/taco_yolo_fr/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    workers=2,
    device="cpu",
    lr0=0.001,
    val=True,
    cache=True,
    exist_ok=True,
    project="runs/detect",
    name="best"
)
