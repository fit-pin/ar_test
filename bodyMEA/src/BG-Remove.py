import os
from rembg import remove, new_session

import cv2 as cv


# 테스트 이미지
IMG = "res/test.jpg"

# 저장경로
SAVE_PATH = "res/background.jpg"

session = new_session("u2net_human_seg")

img = cv.imread(IMG)

result_img = remove(
    img,
    alpha_matting=True,
    alpha_matting_foreground_threshold=20,
    alpha_matting_background_threshold=1,
    alpha_matting_erode_size=1,
    bgcolor=(255, 255, 255, 255),
    session=session,
)

cv.imwrite(SAVE_PATH, result_img)
