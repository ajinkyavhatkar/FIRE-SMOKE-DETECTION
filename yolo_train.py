import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="yolo_dataset/data.yaml",
    epochs=5,
    imgsz=320,          # reduced from 640 — much lighter on memory
    batch=4,             # reduced from 16 — much lighter on memory
    workers=0,
    name="fire_smoke_yolo_test3"
)

print("Training complete!")