import torch
import torch.nn as nn
import scipy.io as sio
import sys
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

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

