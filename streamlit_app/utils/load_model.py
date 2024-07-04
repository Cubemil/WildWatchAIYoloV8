from yolov8 import YOLO

model_path = "../../models/wildwatchyolov8_v2_finetuning05.pt"

def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model
