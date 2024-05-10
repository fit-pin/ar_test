# 모션 트래킹 코드
from typing import Any
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

from torch import Tensor
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from ultralytics.engine.results import Results

BODY_PARTS = {"코": 0, "오른쪽 눈": 1, "왼쪽 눈": 2, "오른쪽 귀": 3, "왼쪽 귀": 4,
              "오른쪽 어깨": 5, "왼쪽 어깨": 6, "오론쪽 팔꿈치": 7, "왼쪽 팔꿈치": 8, "오른쪽 손목": 9,
              "왼쪽 손목": 10, "오른쪽 골반": 11, "왼쪽 골반": 12, "오른쪽 무릎": 13, "왼쪽 무릎": 14,
              "오른쪽 발": 15, "왼쪽 발": 16}

# 상체 점 연결
PARES_TOP = {"팔 길이": ["왼쪽 어깨", "왼쪽 팔꿈치", "왼쪽 손목"], "어께너비": ["왼쪽 어깨", "오른쪽 어깨"], "상체너비": ["왼쪽 어깨", "왼쪽 골반"]}
# 하체 점 연결
PARES_BOTTOM = {"다리 길이": ["왼쪽 골반", "왼쪽 무릎", "왼쪽 발"]}

# 테스트 이미지
img = cv.imread("bodyMEA/test.jpg")

# 점들간의 길이 구하는 함수
def distance(points: list[tuple[float]]):
    distance = 0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

# 기준 사물 높이 가지고 다른 사이즈 예측
def findRealSize(refSize: int, refPx, findPx):
    cm_per_px = refSize / refPx
    return round(findPx * cm_per_px)

# 사람 영역 평가
def personArea(img: cv.typing.MatLike, modelResult: Results):
    # 한 사람만 선택
    personBox = modelResult.boxes[0].xyxy[0]
    annotator = Annotator(img)
    annotator.box_label(personBox, "", (0, 255, 0), (0, 0, 0))
    return annotator.result()

# 사람 키 기준으로 측정하는 함수
def heightToPoint(modelResult: Results, testDist: list[Tensor]):
    # 내 키입력
    MY_KEY = 174

    # 한 사람만 선택
    person = modelResult.boxes[0].xywh[0]

    # 픽셀상 사람 길이
    personPx = int(person[3])

    # 구하려는 사이 픽셀 길이
    dist = distance(testDist)

    # 실제 길이 구하기
    realSize = findRealSize(MY_KEY, personPx, dist)

    print(f"픽셀길이: 구할려는거 {int(dist)}px, 픽셀상 사람 길이 {personPx}px")
    print(f"실제 길이: {realSize}cm")

# 평균 눈 사이 거리로 측정하는 함수
def eyeToPoint(modelResult: Results, testDist: list[Tensor]):
    # 성인 평균 눈 사이 거리
    EYE_TO_DIST = 7.5

    # 한 사람만 선택
    person = modelResult.boxes[0].xywh[0]

    # 픽셀상 사람 길이
    personPx = int(person[3])

    # 눈 사이 픽셀상에 거리
    eyePx = distance(testDist)

    # 실제 길이 구하기
    realSize = findRealSize(EYE_TO_DIST, eyePx, personPx)

    print(f"눈사이 픽셀길이: {int(eyePx)}px, 픽셀상 사람 길이: {personPx}px")
    print(f"실제 사람 길이 길이: {realSize}cm")

# 포즈 구하는
def pose(img: cv.typing.MatLike, modelSrc: str):
    model = YOLO(modelSrc)
    result: Results = model.predict(img)[0]

    # 사람 감지 안되면 예외
    assert (len(result.boxes.cls))

    # 여러 사람 감지 될 시 한사람만 되게
    person1Pose = result.keypoints.xy[0]

    # 대충 길이 테스트해 볼꺼
    lenTest = [
        person1Pose[BODY_PARTS["왼쪽 어깨"]],
        person1Pose[BODY_PARTS["왼쪽 팔꿈치"]],
        person1Pose[BODY_PARTS["왼쪽 손목"]]
    ]

    # 눈 사이 거리
    eyeTest = [
        person1Pose[BODY_PARTS["오른쪽 눈"]],
        person1Pose[BODY_PARTS["왼쪽 눈"]]
    ]

    # 사람 영역 구하기
    img = personArea(img, result)

    # 사람 키 기준으로 측정하기
    # heightToPoint(result, lenTest)

    # 평균 눈사이 거리로 측정하기
    eyeToPoint(result, eyeTest)

    point1 = eyeTest[0]
    point2 = eyeTest[1]

    # 시각화
    for key in BODY_PARTS.keys():
        points = person1Pose[BODY_PARTS[key]]
        cv.putText(img, str(BODY_PARTS[key]), (int(points[0]), int(points[1])),
                   cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2, cv.LINE_4)
        cv.ellipse(img, (int(points[0]), int(points[1])), (8, 8), 0, 0, 360, (0, 0, 255), cv.FILLED)

    return img


res = pose(img, "model/yolov8n-pose.pt")
colorImg = cv.cvtColor(res, cv.COLOR_BGR2RGB)
# cv.imwrite("bodyMEA/result.jpg", colorImg)
plt.imshow(colorImg)
plt.show()
