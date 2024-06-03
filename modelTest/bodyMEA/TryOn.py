import math
import cv2 as cv
import cvzone
import matplotlib.pyplot as plt

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

BODY_PARTS = {
    "코": 0,
    "오른쪽 눈": 1,
    "왼쪽 눈": 2,
    "오른쪽 귀": 3,
    "왼쪽 귀": 4,
    "오른쪽 어깨": 5,
    "왼쪽 어깨": 6,
    "오른쪽 팔꿈치": 7,
    "왼쪽 팔꿈치": 8,
    "오른쪽 손목": 9,
    "왼쪽 손목": 10,
    "오른쪽 골반": 11,
    "왼쪽 골반": 12,
    "오른쪽 무릎": 13,
    "왼쪽 무릎": 14,
    "오른쪽 발": 15,
    "왼쪽 발": 16,
}

MODEL = "model/yolov8n-pose.pt"
PERSON_IMG = "bodyMEA/res/test.jpg"
CLOTHES_IMG = "bodyMEA/res/clothes_result.png"


# reWidth 값 기준으로 사이즈 줄이기
def reSize(img: cv.typing.MatLike, reWidth: int):
    height, width = img.shape[:2]
    reHeight = int(height * reWidth / width)
    return cv.resize(img, (reWidth, reHeight), interpolation=cv.INTER_AREA)


# 채형 사진에 의류 이미지 합성하기
def overlayClothes(backGround: cv.typing.MatLike, clothes: cv.typing.MatLike, personPose):
    # 의류 이미지 가로 실제 보정 배율
    WIDTH_CORR = 2.1

    # X 좌표 보정
    X_POINT_CORR = 0.79

    # Y 좌표 보정
    Y_POINT_CORR = 0.93

    point1 = personPose[BODY_PARTS["왼쪽 어깨"]]
    point2 = personPose[BODY_PARTS["오른쪽 어깨"]]

    # 어깨와 어꺠 사이로 이미지 사이즈 보정
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distance *= WIDTH_CORR

    # 이미지 크기 보정하기
    resize_clothes = reSize(clothes, int(distance))

    # 체형 이미지와 보정된 의류 이미지 합성
    return cvzone.overlayPNG(backGround, resize_clothes, (int(x1 * X_POINT_CORR), int(y1 * Y_POINT_CORR)))


# 사람 이미지 불러오기
personimg = cv.imread(PERSON_IMG)

# 투명도까지 포함된 4채널로
clothesimg = cv.imread(CLOTHES_IMG, cv.IMREAD_UNCHANGED)

model = YOLO("model/yolov8n-pose.pt")
result: Results = model.predict(personimg)[0]

# 사람 감지 안되면 예외
assert len(result.boxes.cls)

# 여러 사람 감지 될 시 한사람만 되게
person1Pose = result.keypoints.xy[0]

# 사람 이미지 투명도 값 추가
personimg_bgra = cv.cvtColor(personimg, cv.COLOR_BGR2BGRA)

# 신체 이미지와 의류 이미지 합성
overLayImg = overlayClothes(personimg_bgra, clothesimg, person1Pose)

# 키포인트 시각화
""" for inedx, point in enumerate(person1Pose):
    cv.ellipse(overLayImg, (int(point[0]), int(point[1])), (8, 8), 0, 0, 360, (0, 0, 255), cv.FILLED)
    cv.putText(overLayImg, str(inedx), (int(point[0]), int(point[1])), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) """

colorImg = cv.cvtColor(overLayImg, cv.COLOR_BGR2RGB)

cv.imwrite("bodyMEA/res/TryOn_reslts.jpg", cv.cvtColor(colorImg, cv.COLOR_RGB2BGR))
plt.imshow(colorImg)
plt.show()
