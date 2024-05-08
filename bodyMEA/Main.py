# 모션 트래킹 코드
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from ultralytics.engine.results import Results

BODY_PARTS = {"코": 0, "오른쪽 눈": 1, "왼쪽 눈": 2, "오른쪽 귀": 3, "왼쪽 귀": 4,
              "오른쪽 어깨": 5, "왼쪽 어깨": 6, "오론쪽 팔꿈치": 7, "왼쪽 팔꿈치": 8, "오른쪽 손목": 9,
              "왼쪽 손목": 10, "오른쪽 골반": 11, "왼쪽 골반": 12, "오른쪽 무릎": 13, "왼쪽 무릎": 14,
              "오른쪽 발": 15, "왼쪽 발": 16}

# 테스트 이미지
img = cv.imread("bodyMEA/test.jpg")
# 내 키
myKey = 174

# 점들간의 길이 구하는 함수
def distance(points: list[tuple[float]]):
    distance = 0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

# 기준 사물 높이 가지고 다른 사이즈 예측
def __findRealSize(refSize: int, refPx, findPx):
    cm_per_px = refSize / refPx
    return round(findPx * cm_per_px)

# 사람 영역 평가
def personArea(img: cv.typing.MatLike, modelResult: Results):
    # 한 사람만 선택
    personBox = modelResult.boxes[0].xyxy[0]
    annotator = Annotator(img)
    annotator.box_label(personBox, "", (0, 255, 0), (0, 0, 0))
    return annotator.result()

# 픽셀상 사람 길이
def personHight(modelResult: Results):
    # 한 사람만 선택
    person = modelResult.boxes[0].xywh[0]
    return int(person[3])

# 포즈 구하는
def pose(img: cv.typing.MatLike, modelSrc: str):
    model = YOLO(modelSrc)
    result: Results = model.predict(img)[0]

    # result.masks 가 None 이면 이미지에 사람이 없는거
    assert (not result.masks)

    # 여러 사람 감지 될 시 한사람만 되게
    person1Pose = result.keypoints.xy[0]

    # 대충 길이 테스트해 볼꺼
    lenTest = [
        person1Pose[BODY_PARTS["왼쪽 어깨"]],
        person1Pose[BODY_PARTS["왼쪽 팔꿈치"]],
        person1Pose[BODY_PARTS["왼쪽 손목"]]
    ]

    # 사람 영역 구하기
    img = personArea(img, result)

    # 구하려는 사이 픽셀 길이
    dist = distance(lenTest)

    # 픽셀상 사람 길이
    personPx = personHight(result)

    # 실제 길이 구하기
    realSize = __findRealSize(myKey, personPx, dist)

    print(f"픽셀길이: 구할려는거 {int(dist)}px, 픽셀상 사람 길이 {personPx}px")
    print(f"실제 길이: {realSize}cm")

    point1 = lenTest[0]
    point2 = lenTest[1]
    point3 = lenTest[2]

    # 시각화
    cv.ellipse(img, (int(point1[0]), int(point1[1])), (8, 8), 0, 0, 360, (0, 0, 255), cv.FILLED)
    cv.ellipse(img, (int(point2[0]), int(point2[1])), (8, 8), 0, 0, 360, (0, 0, 255), cv.FILLED)
    cv.ellipse(img, (int(point3[0]), int(point3[1])), (8, 8), 0, 0, 360, (0, 0, 255), cv.FILLED)

    return img


res = pose(img, "model/yolov8n-pose.pt")
colorImg = cv.cvtColor(res, cv.COLOR_BGR2RGB)
# cv.imwrite("bodyMEA/result.jpg", colorImg)
plt.imshow(colorImg)
plt.show()
