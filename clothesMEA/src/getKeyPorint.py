import torch

from HRnet import pose_hrnet

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("gpu 사용 가능")    
    
model = pose_hrnet.get_pose_net()