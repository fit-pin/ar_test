# 모션 트래킹 코드
from typing import Any, Literal
import cv2 as cv
import matplotlib.pyplot as plt
import math

from torch import Tensor
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
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

# 상체 점 연결
PARES_TOP = {
    "왼쪽 팔 길이": ["왼쪽 어깨", "왼쪽 팔꿈치", "왼쪽 손목"],
    "오른쪽 팔 길이": ["오른쪽 어깨", "오른쪽 팔꿈치", "오른쪽 손목"],
    "어께너비": ["왼쪽 어깨", "오른쪽 어깨"],
    "상체너비": ["왼쪽 어깨", "왼쪽 골반"],
}

# 하체 점 연결
PARES_BOTTOM = {
    "왼쪽 다리 길이": ["왼쪽 골반", "왼쪽 무릎", "왼쪽 발"],
    "오른쪽 다리 길이": ["오른쪽 골반", "오른쪽 무릎", "오른쪽 발"],
}

# 길이 알고리즘 설정
REASLT_MODE: Literal["눈 사이", "키"] = "키"

# 테스트 이미지
img = cv.imread("res/test.jpg")


# reWidth 값 기준으로 사이즈 줄이기
def reSize(img: cv.typing.MatLike, reWidth: int):
    height, width = img.shape[:2]
    reHeight = int(height * reWidth / width)
    return cv.resize(img, (reWidth, reHeight), interpolation=cv.INTER_AREA)


# 점들간의 길이 구하는 함수
def distance(points: list[tuple[float]]):
    distance = 0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


# 기준 사물 높이 가지고 다른 사이즈 예측
def findRealSize(refSize: int, refPx, findPx):
    cm_per_px = refSize / refPx
    return findPx * cm_per_px


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
    personPx = float(person[3])

    # 구하려는 사이 픽셀 길이
    dist = distance(testDist)

    # 실제 길이 구하기
    return findRealSize(MY_KEY, personPx, dist)


# 평균 눈 사이 거리로 측정하는 함수
def eyeToPoint(modelResult: Results, testDist: list[Tensor]):
    # 성인 평균 눈 사이 거리
    EYE_TO_DIST = 6.2

    # 한 사람만 선택
    person1Pose = modelResult.keypoints.xy[0]

    leftEye = person1Pose[BODY_PARTS["왼쪽 눈"]]
    rightEye = person1Pose[BODY_PARTS["오른쪽 눈"]]

    # 눈 사이 픽셀상의 거리
    eyePx = distance([leftEye, rightEye])

    # 구하려는 사이 픽셀 길이
    dist = distance(testDist)

    # 혹시 모르니 사람 키도 구해보는 코드
    person = modelResult.boxes[0].xywh[0]
    personPx = int(person[3])
    realHight = findRealSize(EYE_TO_DIST, eyePx, personPx)
    print(f"예측한 사람 키: {round(realHight, 2)}cm")

    # 실제 길이 구하기
    return findRealSize(EYE_TO_DIST, eyePx, dist)


# 포즈 구하는
def pose(img: cv.typing.MatLike, modelSrc: str):
    model = YOLO(modelSrc)
    result: Results = model.predict(img)[0]

    # 사람 감지 안되면 예외
    assert len(result.boxes.cls)

    # 여러 사람 감지 될 시 한사람만 되게
    person1Pose = result.keypoints.xy[0]

    # 사람 영역 구하기
    img = personArea(img, result)

    # 길이 알고리즘 설정
    resultFunc = heightToPoint
    if REASLT_MODE == "눈 사이":
        resultFunc = eyeToPoint

    shoulder = list(map(lambda x: person1Pose[BODY_PARTS[x]], PARES_TOP["어께너비"]))
    body = list(map(lambda x: person1Pose[BODY_PARTS[x]], PARES_TOP["상체너비"]))

    rightArm = list(map(lambda x: person1Pose[BODY_PARTS[x]], PARES_TOP["오른쪽 팔 길이"]))
    leftArm = list(map(lambda x: person1Pose[BODY_PARTS[x]], PARES_TOP["왼쪽 팔 길이"]))

    rightLeg = list(map(lambda x: person1Pose[BODY_PARTS[x]], PARES_BOTTOM["오른쪽 다리 길이"]))
    leftLeg = list(map(lambda x: person1Pose[BODY_PARTS[x]], PARES_BOTTOM["왼쪽 다리 길이"]))

    shoulderSize = round(resultFunc(result, shoulder), 2)
    bodySize = round(resultFunc(result, body), 2)

    # 왼쪽 오른쪽 비교해서 가장 긴거
    armSize = max([round(resultFunc(result, rightArm), 2), round(resultFunc(result, leftArm), 2)])
    legSize = max([round(resultFunc(result, rightLeg), 2), round(resultFunc(result, leftLeg), 2)])

    print(f"팔 길이: {armSize}cm")
    print(f"어께너비: {shoulderSize}cm")
    print(f"상체너비: {bodySize}cm")
    print(f"다리 길이: {legSize}cm")

    """ 시각화 부분 """
    heght = img.shape[0]
    cv.putText(img, f"armSize: {armSize}cm", (30, int(heght * 0.1)), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
    cv.putText(
        img, f"shoulderSize: {shoulderSize}cm", (30, int(heght * 0.14)), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 221), 2
    )
    cv.putText(img, f"bodySize: {bodySize}cm", (30, int(heght * 0.18)), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    cv.putText(img, f"legSize: {legSize}cm", (30, int(heght * 0.22)), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 162, 0), 2)

    # 상체 시각화
    for name in PARES_TOP.keys():
        if name == "왼쪽 팔 길이" or name == "오른쪽 팔 길이":
            color = (255, 0, 0)
        elif name == "어께너비":
            color = (255, 0, 221)
        else:
            color = (0, 0, 255)

        for i in range(len(PARES_TOP[name]) - 1):
            index = BODY_PARTS[PARES_TOP[name][i]]
            nextIndex = BODY_PARTS[PARES_TOP[name][i + 1]]
            points = person1Pose[index]
            next = person1Pose[nextIndex]
            cv.line(img, (int(points[0]), int(points[1])), (int(next[0]), int(next[1])), color, thickness=cv.LINE_4)

    # 하체 시각화
    for name in PARES_BOTTOM.keys():
        for i in range(len(PARES_BOTTOM[name]) - 1):
            index = BODY_PARTS[PARES_BOTTOM[name][i]]
            nextIndex = BODY_PARTS[PARES_BOTTOM[name][i + 1]]
            points = person1Pose[index]
            next = person1Pose[nextIndex]
            cv.line(
                img, (int(points[0]), int(points[1])), (int(next[0]), int(next[1])), (255, 162, 0), thickness=cv.LINE_4
            )

    return img


img = reSize(img, 700)
res = pose(img, "model/yolov8n-pose.pt")
colorImg = cv.cvtColor(res, cv.COLOR_BGR2RGB)
# cv.imwrite("res/result.jpg", colorImg)
plt.imshow(colorImg)
plt.show()
