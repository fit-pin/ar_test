from ultralytics import YOLO

# Load a model
model = YOLO("model/Clothes-Card.pt")

result =  model.predict("dataset/test/20240930_182425.jpg")[0]

result.save("res/res.jpg")