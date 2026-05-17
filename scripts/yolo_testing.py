"""
YOLOv8 Object Detection on any File

"""

from ultralytics import YOLO

model = YOLO("runs/detect/probe_yolov8n2/weights/best.pt")

model.predict(
    source="videos_probe/probe2.mp4",  # path to your input video, change as needed
    conf=0.2,
    imgsz=640,
    device=0,
    save=True
)