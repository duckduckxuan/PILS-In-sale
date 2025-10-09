from ultralytics import YOLO

model = YOLO("runs/detect/best/weights/best.pt")

model.predict(
    source=0,      # use 0 for webcam
    # source=1,    # use 1 for USB camera
    show=True,
    save=True,
    conf=0.5
)

