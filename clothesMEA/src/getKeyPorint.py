import torch
import cv2
import configuration as con

from HRnet import pose_hrnet
from Utills import Utills

TEST_IMG = "res/test.jpg"
SAVE_IMG = "res/result.jpg"
MODEL_CKP = "model/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth"

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("CUDA 사용 가능")

# 모델 불러오기
model = pose_hrnet.get_pose_net()
model.load_state_dict(torch.load(MODEL_CKP, map_location=device), strict=True)

# to() = cpu() 쓸지 cuda() 쓸지 device 메게 변수로 알아서 처리 
model = torch.nn.DataParallel(model).to(device)
model.eval()
utils = Utills(device)

print("모델 불러오기 성공")

# 이미지 불러오기
img = cv2.imread(TEST_IMG, cv2.IMREAD_COLOR)

#이미지 크기를 288x384 로 변경
reSizeImage = utils.resizeWithPad(img, (288, 384))

# 이미지 정규화 하기
normaImg = utils.getNormalizimage(reSizeImage)

# 해당 __call__  메소드 구현은 부모에 구현되있는데 그쪽에서 forward 함수를 호출하도록 설계함
# 따라서 pose_hrnet.py 에 forward() 함수를 찾아가면 됨
res = model(normaImg)

# 키포인트 추려내기
keyPoints = utils.getKeyPointsResult(res, clothType="반팔")
print("예측 성공")

"""시각화 부분"""
Padding = 2

# 히트맵 사이즈 보정
scaling_factor_x = normaImg.shape[2] / con.HEATMAP_SIZE[0]
scaling_factor_y = normaImg.shape[3] / con.HEATMAP_SIZE[1]

for points in keyPoints[0]:
    joint_x = Padding + points[0] * scaling_factor_x
    joint_y = Padding + points[1] * scaling_factor_y
    if points[0] or points[1]:
        cv2.circle(reSizeImage, (int(joint_x), int(joint_y)), 2, [0, 255, 0], 2)

cv2.imwrite(SAVE_IMG, reSizeImage)
