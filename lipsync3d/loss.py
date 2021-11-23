import sys
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import torch
import torch.nn as nn
import scipy.io as sio
from renderer.face_model import FaceModel
from lib.mesh_io import read_obj
from gcn_util.utils import init_sampling
from lipsync3d.utils import get_downsamp_trans
import os
import numpy as np

class RMSLoss(nn.Module):
    def __init__(self):
        super(RMSLoss, self).__init__()
    
    def forward(self, input, target):
        return torch.sqrt(((input - target)**2).mean())


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
    
    def forward(self, input, target):
        l2_loss = (target - input) ** 2
        l2_loss = torch.mean(l2_loss)

        return l2_loss

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
    
    def forward(self, input, target):
        
        return (torch.abs(input - target)).mean()

class WeightedMSELoss(nn.Module):
    def __init__(self, opt):
        super(WeightedMSELoss, self).__init__()
        downsamp_trans = get_downsamp_trans()

        mat_data = sio.loadmat(opt.matlab_data_path)
        geo_mean = torch.from_numpy(mat_data['geo_mean']).reshape(-1, 3)
        exp_base = torch.from_numpy(mat_data['exp_base'])
        geo_mean = torch.mm(downsamp_trans, geo_mean)

        mouth_vertex_list = []
    
        for i in range(geo_mean.shape[0]):  # 2232
            if geo_mean[i, 1] < 0.:  # y coordinate
                mouth_vertex_list.append(i)

        weight = torch.ones((downsamp_trans.shape[0]))
        weight[mouth_vertex_list] = 50.
        self.weight = weight.reshape(-1, 1).to(opt.device)

    
    def forward(self, input, target):
        
        return (self.weight * (input - target) ** 2).mean()

class calculate_geoLoss(nn.Module):
    def __init__(self, opt):
        super(calculate_geoLoss, self).__init__()
        self.device = opt.device

        self.mat_data = sio.loadmat(opt.matlab_data_path)
        self.id_base = torch.from_numpy(self.mat_data['id_base']).unsqueeze(0).expand(opt.batch_size, -1, -1).to(self.device)
        self.exp_base = torch.from_numpy(self.mat_data['exp_base']).unsqueeze(0).expand(opt.batch_size, -1, -1).to(self.device)
        self.landmark_index = torch.tensor([
            27440, 27208, 27608, 27816, 35472, 34766, 34312, 34022, 33838, 33654,
            33375, 32939, 32244, 16264, 16467, 16888, 16644, 31716, 31056, 30662,
            30454, 30288, 29549, 29382, 29177, 28787, 28111,  8161,  8177,  8187,
            8192,  9883,  9163,  8204,  7243,  6515, 14066, 12383, 11353, 10455,
            11492, 12653,  5828,  4920,  3886,  2215,  3640,  4801, 10795, 10395,
            8935,  8215,  7495,  6025,  5522,  6915,  7636,  8236,  8836,  9555,
            10537,  9064,  8223,  7384,  5909,  7629,  8229,  8829
            ], device=self.device)
        
        self.landmark_weight = torch.tensor([[
                1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                1.0,  1.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
                50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
        ]], device=self.device).unsqueeze(-1)
    
    def forward(self, alpha, delta, reference_alpha, delta_pred):
        geo_gt = self.id_base.bmm(alpha) + self.exp_base.bmm(delta)
        geo_gt = geo_gt.reshape(-1, 35709, 3)[:, self.landmark_index]

        geo_pred = self.id_base.bmm(reference_alpha) + self.exp_base.bmm(delta_pred)
        geo_pred = geo_pred.reshape(-1, 35709, 3)[:, self.landmark_index]

        l2_loss = (geo_gt - geo_pred) ** 2
        l2_loss = torch.sum(self.landmark_weight * l2_loss) / 68.0

        return l2_loss

    # def forward(self, alpha, delta, reference_alpha, delta_pred):
    #     geo_gt = self.id_base.bmm(alpha) + self.exp_base.bmm(delta)
    #     geo_gt = geo_gt.reshape(-1, 35709, 3)
    #     geo_land_gt = geo_gt[:, self.landmark_index]

    #     geo_pred = self.id_base.bmm(reference_alpha).reshape(-1, 35709, 3) + delta_pred
    #     geo_land_pred = geo_pred[:, self.landmark_index]

    #     l2_loss = (geo_gt - geo_pred) ** 2
    #     land_l2_loss = (geo_land_gt - geo_land_pred) ** 2

    #     l2_loss = torch.sum(l2_loss) / 35709.0
    #     land_l2_loss = torch.sum(self.landmark_weight * land_l2_loss) / 68.0
        

    #     return l2_loss, land_l2_loss


class calculate_photo_land_Loss(nn.Module):
    def __init__(self, opt):
        super(calculate_photo_land_Loss, self).__init__()
        self.device = opt.device
        self.face_model = FaceModel(opt=opt, batch_size=opt.batch_size, load_model=True)
        self.photometricLoss = torch.nn.MSELoss()
        self.landmarkLoss = LandmarkLoss(opt)

    def forward(self, alpha, beta, delta, gamma, angle, translation, face_emb, reference_alpha, reference_beta, delta_pred, reference_face_emb, face_landmark_gt):
        
        render_image_gt, _, _, _, _ = self.face_model(alpha, delta, beta, angle, translation, gamma, face_emb, is_train=False)
        render_image_pred, _, landmark_pred, _, _ = self.face_model(reference_alpha, delta_pred, reference_beta, angle, translation, gamma, face_emb, is_train=False)

        return self.photometricLoss(render_image_gt, render_image_pred), self.landmarkLoss(landmark_pred, face_landmark_gt)

class LandmarkLoss(nn.Module):
    def __init__(self, opt):
        super(LandmarkLoss, self).__init__()

        self.device = opt.device

        self.landmark_weight = torch.tensor([[
                1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                1.0,  1.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
                50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
        ]], device=self.device).unsqueeze(-1)

    def forward(self, landmark, landmark_gt):
        landmark_loss = (landmark - landmark_gt) ** 2
        # landmark_loss = torch.sum(self.landmark_weight * landmark_loss) / 68.0
        landmark_loss = torch.mean(self.landmark_weight * landmark_loss)

        return landmark_loss


class FaceembLoss(nn.Module):
    def __init__(self):
        super(FaceembLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, face_emb, face_emb_pred):
        cos = torch.sum(1.0 - self.cos(face_emb, face_emb_pred))

        return cos


class MediapipeLandmarkLoss(nn.Module):
    def __init__(self, opt):
        super(MediapipeLandmarkLoss, self).__init__()
        self.opt = opt
        self.device = opt.device
        self.facemodel = FaceModel(opt, batch_size=opt.batch_size)

        downsamp_trans = get_downsamp_trans()

        mat_data = sio.loadmat(opt.matlab_data_path)
        geo_mean = torch.from_numpy(mat_data['geo_mean']).reshape(-1, 3)
        geo_mean = torch.mm(downsamp_trans, geo_mean)

        mouth_vertex_list = []
    
        for i in range(geo_mean.shape[0]):  # 2232
            if geo_mean[i, 1] < 0.:  # y coordinate
                mouth_vertex_list.append(i)

        weight = torch.ones((478))
        weight[mouth_vertex_list] = 50.
        self.weight = weight.reshape(-1, 1).to(opt.device)

    def forward(self, mediapipe_mesh, geometry_pred, rotation, translation):
        mediapipe_mesh_pred = self.facemodel.geometry_to_pixel(geometry_pred, rotation, translation)

        return torch.mean(self.weight * (mediapipe_mesh - mediapipe_mesh_pred) ** 2)

    
class TemporalLoss(nn.Module):
    def __init__(self):
        super(TemporalLoss, self).__init__()
        self.rms = RMSLoss()

    def forward(self, pred_exp_diff, pred_exp_diff_prv, pred_exp_diff_nxt, target_exp_diff, target_exp_diff_prv, target_exp_diff_nxt):
        loss = self.rms((pred_exp_diff - pred_exp_diff_prv), (target_exp_diff - target_exp_diff_prv)) \
                + self.rms((pred_exp_diff - pred_exp_diff_nxt), (target_exp_diff - target_exp_diff_nxt)) \
                + self.rms((pred_exp_diff_nxt - pred_exp_diff_prv), (target_exp_diff_nxt - target_exp_diff_prv))
        
        return loss
    
    
class emotionLoss(nn.Module):
    def __init__(self):
        super(emotionLoss, self).__init__()
        self.l2 = L2Loss()
        
    def forward(self, mood, index, index_prv, index_nxt):
        loss = self.l2(mood[index], mood[index_prv]) \
                + self.l2(mood[index], mood[index_nxt]) \
                + self.l2(mood[index_nxt], mood[index_prv])
        norm = (mood[index] ** 2).mean()
        return 2 * loss / norm