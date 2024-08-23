import cv2
import numpy as np
import torchvision.transforms as transforms
import torch

W = 288
H = 384

Normal = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def getNormalizimage(src: str):
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    
    # 이미지를 288x384로 크롭하는 코드
    y, x, _ = img.shape
    start_x = x // 2 - int(W) // 2
    start_y = y // 2 - int(H) // 2
    input = cv2.warpAffine(
        img,
        np.float32([[1, 0, -start_x], [0, 1, -start_y]]),  # type: ignore
        (int(W), int(H)),
    )
    nom = Normal(input)
    res = torch.Tensor(np.expand_dims(nom, axis=0))

    return res
