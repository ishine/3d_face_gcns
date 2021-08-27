import os
from collections import OrderedDict

import torch
import torch.nn as nn
from . import networks
import numpy as np
import scipy.io as sio

class Audio2ExpressionNet:
    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self.isTrain = opt.isTrain

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L', 'G_L_ABSOLUTE','G_L_RELATIVE']
        self.exp_ev = self.get_exp_ev()
        self.mat_data = sio.loadmat('renderer/data/data.mat')
        self.exp_base = torch.from_numpy(self.mat_data['exp_base']).to(self.device)
        self.mask = self.get_mouth_mask().to(self.device)
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = []

        # define networks (both generator and discriminator)
        self.net = networks.ExpressionEstimator_Attention(opt.seq_len).to(self.device)

        if self.isTrain:
            # define loss functions
            # self.criterion = networks.WeightedRMSLoss(self.exp_ev)
            self.criterion = networks.MaskedVertexRMSLoss(self.exp_base, self.mask)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    def get_mouth_mask(self):
        mouth_idx = list(set(self.mat_data['mouth_triangles'].reshape(-1, 1).squeeze().tolist()))
        num_vertices = self.exp_base.shape[0] // 3
        mask = torch.zeros(num_vertices)
        mask[mouth_idx] = 1
        mask = torch.stack([mask, mask, mask], dim=1)
        mask = mask + 0.1 * torch.ones_like(mask)

        return mask

    def get_exp_ev(self):
        exp_ev = torch.from_numpy(np.loadtxt('renderer/data/std_exp.txt')[:64])
        exp_ev = torch.sqrt(exp_ev)
        exp_ev = exp_ev / exp_ev[0]
        exp_ev = exp_ev.float()
        return exp_ev.to(self.device)
    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        if self.opt.isTrain:
            self.expressions = input['expressions'].to(self.device)
            self.expressions_prv = input['expressions_prv'].to(self.device)
            self.expressions_nxt = input['expressions_nxt'].to(self.device)

            self.feature_prv = input['feature_prv'].to(self.device)
            self.feature_nxt = input['feature_nxt'].to(self.device)

        self.feature = input['feature'].to(self.device) # b x seq_len x 16 x 29
        self.filename = input['filename']

    def forward(self):
        self.fake_expressions, self.fake_expressions_intermediate = self.net(self.feature)

        if self.opt.isTrain:
            self.fake_expressions_prv, _ = self.net(self.feature_prv)
            self.fake_expressions_nxt, _ = self.net(self.feature_nxt)

    def backward(self):
        self.loss_G_L_ABSOLUTE = 0.0
        self.loss_G_L_RELATIVE = 0.0

        # absolute
        self.loss_G_L_ABSOLUTE += 1000.0 * self.criterion(self.fake_expressions, self.expressions)
        self.loss_G_L_ABSOLUTE += 1000.0 * self.criterion(self.fake_expressions_prv, self.expressions_prv)
        self.loss_G_L_ABSOLUTE += 1000.0 * self.criterion(self.fake_expressions_nxt, self.expressions_nxt)
        self.loss_G_L_ABSOLUTE += 3000.0 * self.criterion(self.fake_expressions_intermediate, self.expressions)

        # relative
        self.loss_G_L_RELATIVE += 20000.0 * self.criterion((self.fake_expressions - self.fake_expressions_nxt), (self.expressions - self.expressions_nxt))
        self.loss_G_L_RELATIVE += 20000.0 * self.criterion((self.fake_expressions_prv - self.fake_expressions), (self.expressions_prv - self.expressions))
        self.loss_G_L_RELATIVE += 20000.0 * self.criterion((self.fake_expressions_prv - self.fake_expressions_nxt), (self.expressions_prv - self.expressions_nxt))
        
        # combine loss and calculate gradients
        self.loss_G_L = self.loss_G_L_ABSOLUTE + self.loss_G_L_RELATIVE
        self.loss_G_L.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def eval(self):
        """Make models eval mode during test time"""
        self.net.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def save_delta(self):
        torch.save(self.fake_expressions[0], os.path.join(self.opt.data_dir, 'reenact_delta', self.filename[0]))

    def update_learning_rate(self):
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save_network(self):
        save_path = os.path.join(self.opt.net_dir, self.opt.net_name_prefix + 'audio2expression_net.pth')
        torch.save(self.net.cpu().state_dict(), save_path)

    def load_network(self):
        load_path = os.path.join(self.opt.net_dir, self.opt.net_name_prefix + 'audio2expression_net.pth')
        state_dict = torch.load(load_path, map_location=self.device)
        self.net.load_state_dict(state_dict)
