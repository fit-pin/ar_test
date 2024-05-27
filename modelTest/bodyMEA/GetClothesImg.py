# 의류 추출
import cv2 as cv
import matplotlib.pyplot as plt

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

# 테스트 이미지
img = cv.imread("bodyMEA/res/clothes.jpg")

# reWidth 값 기준으로 사이즈 줄이기
def reSize(img: cv.typing.MatLike, reWidth: int):
    height, width = img.shape[:2]
    reHeight = int(height * reWidth / width)
    return cv.resize(img, (reWidth, reHeight), interpolation=cv.INTER_AREA)

# 이미지 에서 옷 영역만 추출하기
def getClothesImg(img: cv.typing.MatLike, modelSrc: str):
    model = YOLO(modelSrc)
    result: Results = model.predict(img)[0]
    
    # 의류 부분만 출출하는 마스킹 이미지 만들기
    mask = result.masks[0].data[0].numpy()
    
    height, width = mask.shape
    
    # 완성할 이미지 (완전 투명한 이미지 생성)
    result_img = np.zeros((height, width, 4), dtype=np.uint8)
    
    # 왜인진 모르겠는데 마스크 이미지랑 크기가 달라서 resize
    reSizeImg = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)
    
    # BRRA 이미지 만들기 (투명도가 포함된)
    original_bgra = cv.cvtColor(reSizeImg, cv.COLOR_BGR2BGRA)
    
    for y in range(height):
        for x in range(width):
            # 1로 표현된 것들만 처리
            if mask[y, x] != 0:
                # 해당 영역에 해당하는 오리지널 픽셀 복사해오기
                result_img[y, x] = original_bgra[y, x]
                result_img[y, x][3] = 255 

    return result_img


res = getClothesImg(img, "model/yolov8m-seg.pt")
cv.imwrite("bodyMEA/res/clothes_result.png", res)
plt.imshow(res)
plt.show()
