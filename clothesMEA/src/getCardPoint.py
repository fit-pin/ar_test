import cv2
from matplotlib import pyplot as plt
from torch import Tensor
from ultralytics import YOLO

IMG = "res/test3.jpg"
MODEL = "model/Clothes-Card.pt"


def main():
    model = YOLO(MODEL)
    result = model.predict(IMG)[0]

    if not len(result.obb.cls):
        raise Exception("카드 감지 불가")

    # 예측확율 가장 좋은거 선택
    max_value = 0.0
    max_index = 0
    for i, card in enumerate(result.obb.conf):
        if max_value < float(card):
            max_value = float(card)
            max_index = i
            
    points: Tensor = result.obb.xyxyxyxy[max_index]

    cv_img = cv2.imread(IMG)
    
    [cv2.circle(cv_img, (int(point[0]), int(point[1])), 2, [0, 255, 0], 10) for point in points]

    plt_color = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    plt.imshow(plt_color)
    plt.show()
    
    
if __name__ == "__main__":
    main()