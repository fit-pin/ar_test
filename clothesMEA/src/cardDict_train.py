from matplotlib import pyplot as plt
from ultralytics import YOLO
import torch

# Load a model
model = YOLO("model/yolov8x-obb.pt") 


train = model.train(data="dataset/data.yaml", epochs=100)

