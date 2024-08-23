import torch
import matplotlib.pyplot as pyplot
import cv2

from HRnet import pose_hrnet
import Utills

TEST_IMG = "res/test.jpg"
SAVE_IMG = "res/result.jpg"
MODEL_CKP = "model/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("gpu 사용 가능")
else:
    device = torch.device("cpu")
    print("cpu 사용")

    
model = pose_hrnet.get_pose_net()
model.load_state_dict(torch.load(MODEL_CKP), strict=True)
model = torch.nn.DataParallel(model).cuda()
print("모델 불러오기 성공")

model.eval()

img =  Utills.getNormalizimage("res/test.jpg")
res = model(img)

# cv2.imwrite(SAVE_IMG, img)