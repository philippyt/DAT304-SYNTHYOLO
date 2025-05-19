import subprocess
from ultralytics import YOLO

def train_yolov5(data_yaml, name, project, epochs=100, batch=16, weights="yolov5s.pt"):
    subprocess.run([
        "python", "yolov5/train.py",
        "--img", "640",
        "--batch", str(batch),
        "--epochs", str(epochs),
        "--data", data_yaml,
        "--weights", weights,
        "--name", name,
        "--project", project
    ])

def train_yolovX(model_path, data_yaml, name, project, epochs=100, batch=16):
    model = YOLO(model_path)
    model.train(
        data=data_yaml,
        imgsz=640,
        batch=batch,
        epochs=epochs,
        name=name,
        project=project
    )