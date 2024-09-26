from matplotlib import pyplot as plt
from ultralytics import YOLO

# Load a model
model = YOLO("model/yolov8m.pt")

result =  model.predict("res/test4.jpg")[0]

result.save("res/res.jpg")

plt.show()