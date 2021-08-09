import torch
from torch import nn
from torch.nn import functional as F
import utils
import os
import numpy as np
from lib.mesh_io import read_obj
from .conv import Conv2d
from renderer.mesh_refiner import ChebResBlock
import scipy.io as sio

class SyncNet(nn.Module):
    def __init__(self, opt):
        super(SyncNet, self).__init__()
        self.device = opt.device
        self.mat_data = sio.loadmat(opt.matlab_data_path)
        self.exp_base = torch.from_numpy(self.mat_data['exp_base']).unsqueeze(0).to(self.device)

        self.refer_mesh = read_obj(os.path.join('renderer', 'data', 'bfm09_face_template.obj'))
        self.laplacians, self.downsamp_trans, self.upsamp_trans, self.pool_size = utils.init_sampling(
        self.refer_mesh, os.path.join('renderer', 'data', 'params', 'bfm09_face'), 'bfm09_face')
        self.laplacians = [(torch.FloatTensor(np.array(laplacian.todense())) - torch.diag(torch.ones(laplacian.shape[0]))).to_sparse().to(self.device) for laplacian in self.laplacians]
        self.downsamp_trans = [torch.FloatTensor(np.array(downsamp_tran.todense())).to_sparse().to(self.device) for downsamp_tran in self.downsamp_trans]
        self.K = 6
        self.relu = nn.ReLU()

        self.encoderF = [15, 16, 16, 16, 32]
        self.encoder_fc = nn.Linear(self.pool_size[-1] * self.encoderF[-1], 512, bias=True)

        nn.init.xavier_normal_(self.encoder_fc.weight, gain=1.0)
        nn.init.zeros_(self.encoder_fc.bias)

        self.coef_encoder_cheb_layers = nn.ModuleList([ChebResBlock(self.encoderF[0], self.encoderF[1], self.laplacians[0], self.K),
                                                ChebResBlock(self.encoderF[1], self.encoderF[2], self.laplacians[1], self.K),
                                                ChebResBlock(self.encoderF[2], self.encoderF[3], self.laplacians[2], self.K),
                                                ChebResBlock(self.encoderF[3], self.encoderF[4], self.laplacians[3], self.K)])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def poolwT(self, inputs, L):
        Mp = L.shape[0]
        B, M, Fin = inputs.shape
        B, M, Fin = int(B), int(M), int(Fin)
        x = inputs.permute(1, 0, 2)
        x = torch.reshape(x, (M, B * Fin))
        x = torch.mm(L, x)  # Mp, B*Fin
        x = torch.reshape(x, (Mp, B, Fin))
        x = x.permute(1, 0, 2)

        return x

    
    def coef_encoder(self, inputs):
        B, T, Fin = inputs.shape # inputs : B x T x 64
        inputs = inputs.reshape((B * T, Fin, 1))
        exp_base = self.exp_base.expand(B*T, -1, -1)
        exp = exp_base.bmm(inputs) # exp : B*T x 107127
        exp = exp.reshape(B, T, -1, 3) # exp : B x T x 35709 x 3
        exp = exp.permute(0, 2, 3, 1) # exp : B x 35709 x 3 x T
        exp = exp.reshape(B, -1, 3 * T) # exp : B x 35709 x 3*T
        layer1 = self.coef_encoder_cheb_layers[0](exp)          # layer1 = B x 35709 x 16
        layer2 = self.poolwT(layer1, self.downsamp_trans[0])    # layer2 = B x 8928 x 16
        layer2 = self.coef_encoder_cheb_layers[1](layer2)       # layer2 = B x 8928 x 16
        layer3 = self.poolwT(layer2, self.downsamp_trans[1])    # layer3 = B x 2232 x 16
        layer3 = self.coef_encoder_cheb_layers[2](layer3)       # layer3 = B x 2232 x 16
        layer4 = self.poolwT(layer3, self.downsamp_trans[2])    # layer4 = B x 558 x 16
        layer4 = self.coef_encoder_cheb_layers[3](layer4)       # layer4 = B x 558 x 32
        layer5 = self.poolwT(layer4, self.downsamp_trans[3])    # layer5 = B x 140 x 32
        layer5 = layer5.reshape(B, -1)
        outputs = self.encoder_fc(layer5)   # outputs = B x 512
        outputs = self.relu(outputs)
        return outputs

    def forward(self, audio_sequences, coef_sequences): # audio_sequences := (B, 1, 80, 16) # coef_sequences : (B, T, 64)
        coef_embedding = self.coef_encoder(coef_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        coef_embedding = coef_embedding.view(coef_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        coef_embedding = F.normalize(coef_embedding, p=2, dim=1)


        return audio_embedding, coef_embedding