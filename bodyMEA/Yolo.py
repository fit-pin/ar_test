import datetime
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics.utils.plotting import Annotator
import matplotlib.pyplot as plt

model = YOLO('yolov8n.pt')
img = cv2.imread("bodyMEA/test.jpg")

results = model.predict(img)

for r in results:
    annotator = Annotator(img)

    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]
        print(box.cls)
        
        adjusted_box = [
            b[0],  # left
            b[1] + 10,  # top
            b[2],  # right
            b[3] - 10   # bottom
        ]
        
        heght = adjusted_box[3] - adjusted_box[1]
        
        annotator.box_label(adjusted_box)
        
cv2.line(img, (10, 10), (10, int(heght.numpy())), (255, 0, 0))
plt.imshow(img)
print(heght)
plt.show()
