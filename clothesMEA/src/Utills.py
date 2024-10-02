import math
import numpy as np
import torchvision.transforms as transforms
from custumTypes import BottomMeaType, TopMeaType, maskKeyPointsType
from torch import device as Device, stack
from torch import Tensor
from torch import tensor
from torch import zeros

from cv2.typing import MatLike
from cv2 import resize, copyMakeBorder, BORDER_CONSTANT

import configuration as con
from typing import Any, Literal, Tuple


Normal = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
"""이미지 정규화 구성"""


maskKeyPoints: dict[maskKeyPointsType, tuple] = {
    "반팔": (0, 25),
    "긴팔": (25, 58),
    "반팔 아우터": (58, 89),
    "긴팔 아우터": (89, 128),
    "조끼": (128, 143),
    "슬링": (143, 158),
    "반바지": (158, 168),
    "긴바지": (168, 182),
    "치마": (182, 190),
    "반팔 원피스": (190, 219),
    "긴팔 원피스": (219, 256),
    "조끼 원피스": (256, 275),
    "슬링 원피스": (275, 294),
}
"""
키포인트 마스크
[예시 참고 사진](https://github.com/switchablenorms/DeepFashion2/blob/master/images/cls.jpg)
"""


TopMea: dict[TopMeaType, tuple] = {
    "어깨너비": (6, 32),
    "가슴단면": (15, 23),
    "소매길이": (32, 31, 30, 29, 28),
    "총장": (0, 19),
}

BottomMea: dict[BottomMeaType, tuple] = {
    "허리단면": (0, 2),
    "밑위": (1, 8),
    "엉덩이단면": (3, 13),
    "허벅지단면": (8, 13),
    "총장": (0, 3, 4, 5),
    "밑단단면": (5, 6),
}


class Utills:
    def __init__(self, device: Device):
        """
        Args:
            device (torch.device): 연산 gpu 장치
        """
        self.device = device

    def getNormalizimage(self, img: MatLike):
        """
        이미지를 정규화 하여 반환 합니다.</br>
        반환된 이미지로 `getKeyPointsResult()` 함수를 호출하여 keyPoint 를 예측합니다

        Args:
            img (cv2.typing.MatLike): cv 이미지

        Returns:
            Tensor: 정규화된 이미지 텐서
        """
        nom = Normal(img)
        res = Tensor(np.expand_dims(nom, axis=0))
        return res

    def getKeyPointsResult(
        self,
        predOutput: Tensor,
        is_mask=True,
        flipTest=False,
        clothType: maskKeyPointsType = "반팔",
    ):
        """
        정규화된 이미지로 키포인트를 얻습니다

        Args:
            predOutput (Tensor): 정규화 이미지 텐서
            is_mask (bool, optional): 의류 타입별 채널 마스크 여부 (안하면 오차 keyPoint가 발생)
            flipTest (bool, optional): flipTest 여부
            clothType (int, optional): 의류타입

        Returns:
            Tensor: 의류 키포인트
        """

        if is_mask:
            channel_mask = zeros((1, 294, 1)).to(self.device).float()

            rg = maskKeyPoints[clothType]
            index = (
                tensor(
                    [list(range(rg[0], rg[1]))],
                    device=channel_mask.device,
                    dtype=channel_mask.dtype,
                )
                .transpose(1, 0)
                .long()
            )
            channel_mask[0].scatter_(0, index, 1)

            predOutput = predOutput * channel_mask.unsqueeze(3)

        preds_local = self.__get_final_preds(predOutput.detach().cpu().numpy())

        return preds_local

    def resizeWithPad(
        self,
        image: MatLike,
        new_shape: Tuple[int, int],
        padding_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> tuple[MatLike, dict[Literal["top", "bottom", "left", "right"], int]]:
        """
        비율을 유지하여 이미지를 자릅니다.</br>
        이때 비율을 유지하기 위해 잘려진 부분은 `padding_color`로 채워 집니다

        Params:
            image (MatLike): 원본 이미지
            new_shape (Tuple[int, int]): 바꿀 크기
            padding_color (Tuple[int, int, int]): 잘려진 부분 색상
        Returns:
            tuple[cv2.typing.MatLike, dict[str, int]]</br>
            - `[0]`: 잘려진 이미지
            - `[1]`: paddig 값
        """
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(new_shape)) / max(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])

        test_w = new_shape[0] - new_size[0]
        test_h = new_shape[1] - new_size[1]

        if test_w < 0:
            ratio = float(new_shape[0] / new_size[0])
            new_size = tuple([int(x * ratio) for x in new_size])
        elif test_h < 0:
            ratio = float(new_shape[1] / new_size[1])
            new_size = tuple([int(x * ratio) for x in new_size])

        image = resize(image, new_size)

        delta_w = new_shape[0] - new_size[0]
        delta_h = new_shape[1] - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        image = copyMakeBorder(
            image, top, bottom, left, right, BORDER_CONSTANT, value=padding_color
        )

        return image, {"top": top, "bottom": bottom, "left": left, "right": right}

    # 이 함수는 원본에서 불러온 것
    def __get_max_preds(self, batch_heatmaps):
        assert isinstance(
            batch_heatmaps, np.ndarray
        ), "batch_heatmaps should be numpy.ndarray"
        assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"

        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds

    # 이 함수는 원본에서 불러온 것
    def __get_final_preds(self, output):
        heatmap_height = con.HEATMAP_SIZE[0]
        heatmap_width = con.HEATMAP_SIZE[1]

        batch_heatmaps = output
        coords = self.__get_max_preds(batch_heatmaps)

        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px],
                        ]
                    )
                    coords[n][p] += np.sign(diff) * 0.25

        return coords

    def getMEApoints(
        self, resultPoint: Tensor, type: Literal["긴팔", "긴바지"]
    ) -> dict[Any, Tensor]:
        """
        실측 크기에 유의미한 좌표값을 얻습니다.</br>
        
        긴팔 Type: `dict[TopMeaType, list[Tensor]]`</br>
        긴바지 Type: `dict[BottomMeaType, list[Tensor]]`</br>
        Args:
            resultPoint (Tensor): 전체 결과 키포인트
            type (Literal["긴팔", "긴바지"]):

        Returns:
            dict[Any, Tensor]: 결과반환
        """
        resultDict = {}
        if type == "긴팔":
            for key in TopMea.keys():
                # 이게 리스트 안에 Tensor 가 있어서 stack() 함수로 전체를 변환해야함
                resultDict[key] = stack([resultPoint[index] for index in TopMea[key]])
        elif type == "긴바지":
            for key in BottomMea.keys():
                resultDict[key] = stack([resultPoint[index] for index in BottomMea[key]])

        return resultDict

    def distance(self, points: Tensor | list) -> float:
        """여러 점들 사이 길이 구하는 함수

        Args:
            points (list[tuple[float]]): (x,  y) 이걸 구하고 싶은 만큼 list로 묶어서

        Returns:
            float: 점 전체 길이
        """

        distance = 0
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def findRealSize(self, refSize: float, refPx: float, findPx: float):
        """기준 사물 높이 가지고 다른 사이즈 예측

        Args:
            refSize (float): 기준 사물 크기(cm)
            refPx (float): 기준 사물 픽셀상 크기
            findPx (float): 찾으려는 사물 픽셀상 크기

        Returns:
            float: 사물사이즈(cm)
        """

        cm_per_px = refSize / refPx
        return findPx * cm_per_px
