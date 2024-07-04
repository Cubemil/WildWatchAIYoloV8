import cv2
import numpy as np

def run_inference(model, image):
    results = model.predict(image)
    return results
