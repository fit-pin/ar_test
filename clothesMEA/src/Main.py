from random import randint
import threading
from typing import Literal
import matplotlib.pyplot as plt

from torch import Tensor, cat, le, load, tensor
from torch import device as Device
from torch.nn import DataParallel
from torch.cuda import is_available

import cv2
from ultralytics import YOLO
import configuration as con

from HRnet import pose_hrnet

from Utills import TopMeaType, Utills
from custumTypes import BottomMeaType, maskKeyPointsType

TEST_IMG = "res/test3.jpg"
SAVE_IMG = "res/result.jpg"
CLOTH_TYPE: maskKeyPointsType = "긴팔"

KEYPOINT_MODEL_CKP = "model/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth"
CARDPOINT_MODEL_CKP = "model/Clothes-Card.pt"


def getCardHeight(syncData: dict, img: cv2.typing.MatLike, utils: Utills):
    print("getCardPoint: 시작")
    model = YOLO(CARDPOINT_MODEL_CKP)
    print("getCardPoint: 모델 불러오기 성공")
    result = model.predict(img)[0]
    print("getCardPoint: 예측 성공")

    if not len(result.obb.cls):  # type: ignore
        raise Exception("카드 감지 불가")

    # 예측확율 가장 좋은거 선택
    max_value = 0.0
    max_index = 0
    for i, card in enumerate(result.obb.conf):  # type: ignore
        if max_value < float(card):
            max_value = float(card)
            max_index = i

    points: Tensor = result.obb.xywhr[max_index]  # type: ignore
    w, h = points[2:4]

    # 제일 작은 값이 세로 이므로
    height = h
    if int(w) < int(h):
        height = w

    syncData["getCardHeight"] = height


def getKeyPoints(syncData: dict, img: cv2.typing.MatLike, utils: Utills):
    print("getKeyPoints: 시작")
    # 모델 불러오기
    model = pose_hrnet.get_pose_net()
    model.load_state_dict(
        load(KEYPOINT_MODEL_CKP, map_location=utils.device), strict=True
    )

    # to() = cpu() 쓸지 cuda() 쓸지 device 메게 변수로 알아서 처리
    model = DataParallel(model).to(utils.device)
    model.eval()
    utils = Utills(utils.device)
    print("getKeyPoints: 모델 불러오기 성공")

    # 이미지 크기를 288x384 로 변경
    reSizeImage, padding = utils.resizeWithPad(img, (288, 384))

    print(f"getKeyPoints: 이미지에 적용된 패딩: {padding}")

    # 이미지 정규화 하기
    normaImg = utils.getNormalizimage(reSizeImage)

    # 해당 __call__  메소드 구현은 부모에 구현되있는데 그쪽에서 forward 함수를 호출하도록 설계함
    # 따라서 pose_hrnet.py 에 forward() 함수를 찾아가면 됨
    res = model(normaImg)

    # 키포인트 추려내기
    keyPoints = utils.getKeyPointsResult(res, clothType=CLOTH_TYPE)
    print("getKeyPoints: 예측 성공")

    """시각화 부분"""
    pointPadding = 2

    # 히트맵 사이즈 보정
    scaling_factor_x = con.IMG_SIZE[1] / con.HEATMAP_SIZE[0]
    scaling_factor_y = con.IMG_SIZE[0] / con.HEATMAP_SIZE[1]

    result_points = Tensor()
    for points in keyPoints[0]:
        # 288x384 에서 표시되야 하는 점
        joint_x = pointPadding + points[0] * scaling_factor_x
        joint_y = pointPadding + points[1] * scaling_factor_y

        # 원본 이미지 비율에서 보정 되야 하는 크기
        ratio_x = img.shape[1] / (con.IMG_SIZE[0] - padding["left"] - padding["right"])
        ratio_y = img.shape[0] / (con.IMG_SIZE[1] - padding["top"] - padding["bottom"])
        if points[0] or points[1]:
            # 최종적으로 패딩 값에 따른 점 위치 수정
            final_x = joint_x * ratio_x - (ratio_x * padding["left"])
            final_y = joint_y * ratio_y - (ratio_y * padding["top"])

            # 2차원으로 변경
            temp = tensor([final_x, final_y]).unsqueeze(0)
            result_points = cat((result_points, temp))

    syncData["getKeyPoints"] = result_points


# 싩측크기 참고용 시각화 코드
def refKeyPoint(img: cv2.typing.MatLike, resultPoint: Tensor):
    for i, point in enumerate(resultPoint):
        R = randint(0, 255)
        G = randint(0, 255)
        B = randint(0, 255)
        cv2.circle(img, (int(point[0]), int(point[1])), 2, [B, G, R], 10)
        cv2.putText(
            img,
            str(i),
            (int(point[0]), int(point[1] - 20)),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (B, G, R),
            5,
        )

    cv2.imwrite("res/refTest.jpg", img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def viewSample(img: cv2.typing.MatLike, MEADdata: dict[TopMeaType, list[Tensor]]):
    for part in MEADdata.keys():
        points = MEADdata[part]
        R = randint(0, 255)
        G = randint(0, 255)
        B = randint(0, 255)
        for i, point in enumerate(points):
            cv2.circle(img, (int(point[0]), int(point[1])), 2, [B, G, R], 10)
            if i < len(points) - 1:
                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))
                cv2.line(img, pt1, pt2, [B, G, R], 5)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def main():
    device = Device("cpu")
    if is_available():
        device = Device("cuda:0")
        print("CUDA 사용 가능")

    utils = Utills(device)

    # 이미지 불러오기
    img = cv2.imread(TEST_IMG, cv2.IMREAD_COLOR)

    # 멀티쓰레딩 결과 저장용
    syncData: dict[Literal["getCardHeight"] | Literal["getKeyPoints"], Tensor] = dict()

    # getCardHeight 쓰레드 생성
    CardHeightThread = threading.Thread(
        target=getCardHeight, args=(syncData, img, utils)
    )
    # getKeyPoints 쓰레드 생성
    keyPointThread = threading.Thread(target=getKeyPoints, args=(syncData, img, utils))

    # 쓰레드 시작
    CardHeightThread.start()
    keyPointThread.start()

    # 완료대기
    CardHeightThread.join()
    keyPointThread.join()

    ch_point = syncData["getKeyPoints"]
    card_px = float(syncData["getCardHeight"])

    print(f"감지된 점: {len(ch_point)}개")

    # 긴팔: TopMeaType
    # 긴바지: BottomMeaType
    MEAData: dict[TopMeaType, list[Tensor]] = utils.getMEApoints(ch_point, CLOTH_TYPE)  # type: ignore

    pixelDist_Dict: dict[TopMeaType, float] = {}
    for idx in MEAData.keys():
        pixelDist_Dict[idx] = utils.distance(MEAData[idx])

    realDist_Dict: dict[TopMeaType, float] = {}
    for idx in pixelDist_Dict.keys():
        realDist_Dict[idx] = utils.findRealSize(
            con.CARD_SIZE[1], card_px, pixelDist_Dict[idx]
        )

    print(realDist_Dict)


if __name__ == "__main__":
    main()
