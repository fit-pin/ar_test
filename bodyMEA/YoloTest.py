import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import matplotlib.pyplot as plt

model = YOLO('yolov8n.pt')
img = cv2.imread("bodyMEA/test.jpg")

results = model.predict(img)

annotator = Annotator(img)

findObject = {}

boxes = results[0].boxes
for box in boxes:
    b = box.xyxy[0]
    # print(box.cls)

    adjusted_box = [
        b[0],  # 왼쪽
        b[1],  # 위
        b[2],  # 오른쪽
        b[3]   # 아래
    ]

    findObject.update({
        model.names[int(box.cls)]: adjusted_box
    })

    annotator.box_label(adjusted_box, model.names[int(box.cls)], (0, 255, 0), (0, 0, 0))

# 책 이미지 높이
# bookHight = findObject["book"][3] - findObject["book"][1]

# 사람 높이
personHight = findObject["person"][3] - findObject["person"][1]

# print(bookHight)
print(personHight)

# 기준 사물 높이 가지고 다른 사이즈 예측
def findRealSize(refSize: int, refPx, findPx):
    cm_per_px = refSize / refPx
    return round(findPx * cm_per_px)


# findReal = findRealSize(174, float(personHight), float(bookHight))

# print(f"실제 사이즈: {findReal}cm")

plt.imshow(img)
plt.show()
