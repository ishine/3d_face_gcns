import scipy
import torch
import cv2
import scipy.io as sio
import numpy as np
from skimage import feature
from tqdm import tqdm

class mouth_retrieval():
    def __init__(self, opt, tgt_delta_list, tgt_crop_lip_h, crop_lip_image_list):
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
        self.tgt_crop_lip_h = tgt_crop_lip_h
        self.crop_lip_image_list = crop_lip_image_list
        self.tgt_LBP = self.get_LBP()
    
    
    def get_LBP(self):
        tgt_LBP = []
        for i in tqdm(range(len(self.crop_lip_image_list))):
            tgt_LBP.append(self.calculate_LBP(self.crop_lip_image_list[i], self.tgt_crop_lip_h[i]))
            
        tgt_LBP = torch.cat(tgt_LBP, dim=1)
        return tgt_LBP
    
    
    def calculate_LBP(self, crop_lip_image_path, h):
        img = cv2.imread(crop_lip_image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:h, :]
        lbp_59 = feature.local_binary_pattern(gray_img, 8, 1, method='nri_uniform')
        lbp_16 = feature.local_binary_pattern(gray_img, 4, 1, method='default')
        hist_59 = scipy.stats.itemfreq(lbp_59)
        hist_16 = scipy.stats.itemfreq(lbp_16)
        
        norm_hist_59 = torch.zeros((59))
        norm_hist_16 = torch.zeros((16))
        
        for i in range(len(hist_59)):
            norm_hist_59[int(hist_59[i][0])] = hist_59[i][1]
        
        for i in range(len(hist_16)):
            norm_hist_16[int(hist_16[i][0])] = hist_16[i][1]
        
        norm_hist_59 = norm_hist_59 / sum(norm_hist_59)
        norm_hist_16 = norm_hist_16 / sum(norm_hist_16)
        
        norm_hist = torch.cat([norm_hist_59, norm_hist_16]).reshape(-1, 1)
        
        return norm_hist
    
    
    def chi_square_distance(self, src_LBP):
        numerator = (self.tgt_LBP - src_LBP).pow(2)
        denominator = self.tgt_LBP + src_LBP + 1e-10
        cost = torch.div(numerator, denominator) * 0.5
        cost = cost.sum(dim=0)
        return cost
    
    def retrieve(self, delta, crop_lip_image_path, h):
        delta = delta.clone().cuda()
        expression = self.exp_base.mm(delta).reshape(35709, 3, 1)[self.landmark_index].reshape(-1, 1)
        src_LBP = self.calculate_LBP(crop_lip_image_path, h)
        expr_cost = (self.tgt_expressions - expression).pow(2).sum(dim=0).cpu()
        LBP_cost = self.chi_square_distance(src_LBP)
        index = (4 * expr_cost + LBP_cost).argmin()
        return index