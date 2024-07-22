import os
from rembg import remove, new_session

import cv2 as cv
import matplotlib.pyplot as plt

  
# 테스트 이미지
IMG = "bodyMEA/res/test.jpg"

session = new_session("u2net_human_seg")

img = cv.imread(IMG)

result_img = remove(
    img,
    alpha_matting=True,
    alpha_matting_foreground_threshold=20,
    alpha_matting_background_threshold=1,
    alpha_matting_erode_size=1,
    session=session
)


res =  cv.cvtColor(result_img, cv.COLOR_BGRA2RGBA)
plt.imshow(res)
plt.show()