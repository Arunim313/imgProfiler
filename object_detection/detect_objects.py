import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from readrr import *


def detect_objects(image: np.ndarray, model_name: str = 'yolov8s') -> np.ndarray:
    """
    Apply YOLOv8 object detection on an image (Only supports .jpg format).

    Parameters:
    - image (np.ndarray): The input image as a NumPy array (RGB format).
    - model_name (str, optional): YOLO model variant ('yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'). Defaults to 'yolov8s'.

    Returns:
    - np.ndarray: Image with bounding boxes and labels drawn around detected objects.
    """
    # Load YOLOv8 model
    model = YOLO(model_name)
    results = model(image)
    annotated_img = results[0].plot()

    return annotated_img