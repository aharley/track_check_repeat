import sys
sys.path.append('nets/raft_core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn

from raft import RAFT
from util import InputPadder

class RaftNet(nn.Module):
    def __init__(self, ckpt_name='./nets/raft_models/raft-sintel.pth'):
        super(RaftNet, self).__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument('--small', action='store_true', help='use small model')
        # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--mixed_precision', action='store_false', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args, _ = parser.parse_known_args()

        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(ckpt_name))

    def forward(self, image1, image2, iters=20):
        # input images are in [-0.5, 0.5]
        # raftnet wants the images to be in [0,255]
        image1 = (image1 + 0.5) * 255.0
        image2 = (image2 + 0.5) * 255.0

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        flow_low, flow_up, feat = self.model(image1, image2, iters=iters, test_mode=True)
        flow_up = padder.unpad(flow_up)
    
        return flow_up, feat
