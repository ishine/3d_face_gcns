import torch
import cv2
import scipy.io as sio
import numpy as np

class mouth_retrieval():
    def __init__(self, opt, tgt_delta_list, crop_lip_image_list):
        self.opt = opt
        self.mat_data = sio.loadmat(opt.matlab_data_path)
        self.device = opt.device
        self.exp_base = torch.from_numpy(self.mat_data['exp_base']).to(self.device)
        
        self.landmark_index = torch.tensor([
            27440, 27208, 27608, 27816, 35472, 34766, 34312, 34022, 33838, 33654,
            33375, 32939, 32244, 16264, 16467, 16888, 16644, 31716, 31056, 30662,
            30454, 30288, 29549, 29382, 29177, 28787, 28111,  8161,  8177,  8187,
            8192,  9883,  9163,  8204,  7243,  6515, 14066, 12383, 11353, 10455,
            11492, 12653,  5828,  4920,  3886,  2215,  3640,  4801, 10795, 10395,
            8935,  8215,  7495,  6025,  5522,  6915,  7636,  8236,  8836,  9555,
            10537,  9064,  8223,  7384,  5909,  7629,  8229,  8829
        ], device=self.device)
        
        tgt_deltas = torch.cat(tgt_delta_list, dim=1).to(self.device)
        self.tgt_expressions = self.exp_base.mm(tgt_deltas).reshape(35709, 3, len(tgt_delta_list))[self.landmark_index].reshape(-1, len(tgt_delta_list))
        
        
    def retrieve(self, delta):
        delta = delta.clone().cuda()
        expression = self.exp_base.mm(delta).reshape(35709, 3, 1)[self.landmark_index].reshape(-1, 1)
        cost = (self.tgt_expressions - expression).pow(2).sum(dim=0)
        index = cost.argmin()
        return index