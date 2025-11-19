from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="datasets/taco_yolo_fr/data.yaml",
    epochs=300,
    imgsz=1024,
    batch=8,
    device=0,
    lr0=0.002,
    workers=2,
    cache=False,
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.2,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    project="runs/detect",
    name="best_v2",
    classes=None,
)