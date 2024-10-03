from ultralytics import YOLO

# Load a model
model = YOLO("model/Clothes-Card.pt")

result = model.predict("res/test3.jpg")[0]
print(result.obb)

result.save("res/res.jpg")
