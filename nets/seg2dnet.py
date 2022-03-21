import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")

from utils.basic import print_, print_stats

# from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

EPS = 1e-4
class Seg2dNet(nn.Module):
    def __init__(self, in_dim=3, num_classes=1, imagenet_init=True, shallow=False):
        super(Seg2dNet, self).__init__()

        self.shallow = shallow
        self.num_classes = num_classes

        self.net = lraspp_mobilenet_v3_large(
            pretrained=False,
            progress=True,
            pretrained_backbone=True,
            num_classes=num_classes,
        )

        self.mean_ = torch.from_numpy(np.array((0.485, 0.456, 0.406))).reshape(1, 3, 1, 1).float().cuda()
        self.std_ = torch.from_numpy(np.array((0.229, 0.224, 0.225))).reshape(1, 3, 1, 1).float().cuda()

    def forward(self, rgb):
        total_loss = torch.tensor(0.0).cuda()
        B, C, H, W = rgb.shape

        rgb = rgb + 0.5 # go from [-.5,.5] to [0,1]
        rgb_ = (rgb - self.mean_) / self.std_
        seg_e = self.net(rgb_)['out']
        
        return seg_e
