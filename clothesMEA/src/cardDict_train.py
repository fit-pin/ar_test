from ultralytics import YOLO

# Load a model
model = YOLO("model/yolov8m.pt") 

train = model.train(data="dataset/data.yaml", epochs=100)

