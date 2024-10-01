import threading
import matplotlib.pyplot as plt

from torch import Tensor, load
from torch import device as Device
from torch.nn import DataParallel
from torch.cuda import is_available

import cv2
from ultralytics import YOLO
import configuration as con

from HRnet import pose_hrnet
from Utills import Utills

TEST_IMG = "res/test3.jpg"
SAVE_IMG = "res/result.jpg"

KEYPOINT_MODEL_CKP = "model/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth"
CARDPOINT_MODEL_CKP = "model/Clothes-Card.pt"


def getCardPoint(syncData: dict, img: cv2.typing.MatLike, utils: Utills):
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

    points: Tensor = result.obb.xyxyxyxy[max_index]  # type: ignore

    # list로 바꾸기
    res = [[float(point[0]), float(point[1])] for point in points]
    syncData["getCardPoint"] = res


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

    # 리사이징된 이미지 보기
    # plt.imshow(reSizeImage)
    # plt.show()

    # 이미지 정규화 하기
    normaImg = utils.getNormalizimage(reSizeImage)

    # 해당 __call__  메소드 구현은 부모에 구현되있는데 그쪽에서 forward 함수를 호출하도록 설계함
    # 따라서 pose_hrnet.py 에 forward() 함수를 찾아가면 됨
    res = model(normaImg)

    # 키포인트 추려내기
    keyPoints = utils.getKeyPointsResult(res, clothType="긴팔")
    print("getKeyPoints: 예측 성공")

    """시각화 부분"""
    pointPadding = 2

    # 히트맵 사이즈 보정
    scaling_factor_x = con.IMG_SIZE[1] / con.HEATMAP_SIZE[0]
    scaling_factor_y = con.IMG_SIZE[0] / con.HEATMAP_SIZE[1]

    # 멀티프로세스에서는 tenor 지원 안함
    result_points: list[list[int]] = []
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
            result_points.append([final_x, final_y])

    syncData["getKeyPoints"] = result_points


def main():
    device = Device("cpu")
    if is_available():
        device = Device("cuda:0")
        print("CUDA 사용 가능")

    utils = Utills(device)

    # 이미지 불러오기
    img = cv2.imread(TEST_IMG, cv2.IMREAD_COLOR)

    # 멀티쓰레딩 결과 저장용
    syncData: dict = dict()
    
    # getKeyPoints 쓰레드 생성
    CardPointThread = threading.Thread(target=getCardPoint, args=(syncData, img, utils))
    # getKeyPoints 쓰레드 생성
    keyPointThread = threading.Thread(target=getKeyPoints, args=(syncData, img, utils))

    # 쓰레드 시작
    CardPointThread.start()
    keyPointThread.start()

    # 완료대기
    CardPointThread.join()
    keyPointThread.join()
    
    print(syncData)


    # cv2.circle(img, (int(ch_keyPoints[:][0]), int(ch_keyPoints[:][1])), 2, [0, 255, 0], 15)

    # cv2.imwrite(SAVE_IMG, img)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()


if __name__ == "__main__":
    main()
