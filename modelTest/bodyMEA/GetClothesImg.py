# 의류 추출
import cv2 as cv
import matplotlib.pyplot as plt

from rembg import remove

# 테스트 이미지
img = cv.imread("bodyMEA/res/clothes.jpg")


# reWidth 값 기준으로 사이즈 줄이기
def reSize(img: cv.typing.MatLike, reWidth: int):
    height, width = img.shape[:2]
    reHeight = int(height * reWidth / width)
    return cv.resize(img, (reWidth, reHeight), interpolation=cv.INTER_AREA)


# 이미지 에서 옷 영역만 추출하기
def getClothesImg(img: cv.typing.MatLike):
    # 누끼 따기
    result_img = remove(
        img,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    )

    # 알파 채널 분리
    alpha_channel = result_img[:, :, 3]

    # 투명하지 않은 영역 찾기
    coords = cv.findNonZero(alpha_channel)

    # 투명하지 않은 영역의 바운딩 박스 계산
    x, y, w, h = cv.boundingRect(coords)

    # 바운딩 박스를 기준으로 이미지 자르기
    cropped_image = result_img[y : y + h, x : x + w]

    return cropped_image


res = getClothesImg(img)
cv.imwrite("bodyMEA/res/clothes_result.png", res)
plt.imshow(res)
plt.show()
