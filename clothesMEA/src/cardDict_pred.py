from matplotlib import pyplot as plt
from ultralytics import YOLO

# Load a model
model = YOLO("model/yes_ch2.pt")

result =  model.predict("dataset/test/images/20240930_183633.jpg")[0]

result.save("res/res.jpg")

plt.show()