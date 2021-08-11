import torch
from torch import nn
from collections import OrderedDict
import os
from . import networks
from .syncnet import SyncNet

class Wav2DeltaModel(nn.Module):
    def __init__(self, opt):
        super(Wav2DeltaModel, self).__init__()
        self.opt = opt
        self.device = opt.device
        self.isTrain = opt.isTrain

        self.loss_names = ['Delta', 'Sync']
        self.visual_names = []

        self.net = networks.Wav2Delta(opt).to(self.device)
        self.syncnet = SyncNet(opt)
        pretrained_dir = os.path.join(opt.data_dir, 'syncnet_ckpt')
        self.load_syncnet(pretrained_dir)
        self.syncnet.to(self.device)
        self.syncnet_epoch = opt.syncnet_epoch

        if self.isTrain:
            self.criterionDelta = nn.MSELoss()
            self.criterionSync = networks.SyncLoss(opt.device)

            self.optimizer = torch.optim.Adam([{'params' : self.net.decoder_fc.parameters()}, 
                                                {'params' : self.net.decoder_cheb_layers.parameters()},
                                                {'params' : self.net.last_cheb.parameters()}], lr=opt.lr, betas=(0.5, 0.999))

    def load_syncnet(self, pretrained_dir):
        ckpt_list = sorted(os.listdir(pretrained_dir))
        pretrained_path = os.path.join(pretrained_dir, ckpt_list[-1])

        ckpt = torch.load(pretrained_path, map_location='cpu')["state_dict"]
        self.syncnet.load_state_dict(ckpt)

        for p in self.syncnet.parameters():
            p.requires_grad = False


    def set_input(self, input):
        if self.opt.isTrain:
            self.indiv_mels = input['indiv_mels'].to(self.device)
            self.delta_gt = input['delta_gt'].to(self.device)
        else:
            self.filename = input['filename']

        self.mel = input['mel'].to(self.device)

    def forward(self):
        
        self.delta = self.net(self.indiv_mels)


    def backward(self):
        self.loss_Delta = self.criterionDelta(self.delta, self.delta_gt)
        audio_emb, coef_emb = self.syncnet(self.mel, self.delta)
        self.loss_Sync = self.criterionSync(audio_emb, coef_emb)

        self.loss = self.opt.lambda_sync * self.loss_Sync + (1 - self.opt.lambda_sync) * self.loss_Delta

        self.loss.backward()

    def optimize_parameters(self, epoch):
        if epoch >= self.syncnet_epoch:
            self.opt.lambda_sync = 0.9
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
        torch.save(self.delta[0], os.path.join(self.opt.data_dir, 'reenact_delta', self.filename[0]))

    def update_learning_rate(self):
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save_network(self):
        save_path = os.path.join(self.opt.net_dir, 'delta_net.pth')
        torch.save(self.net.cpu().state_dict(), save_path)

    def load_network(self):
        load_path = os.path.join(self.opt.net_dir, 'delta_net.pth')
        state_dict = torch.load(load_path, map_location=self.device)
        self.net.load_state_dict(state_dict)