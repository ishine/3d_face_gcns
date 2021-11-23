import sys
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import os
import numpy as np
import torch
import torch.nn as nn
from lib.mesh_io import read_obj
from gcn_util.utils import init_sampling
from renderer.mesh_refiner import ChebConv, ChebResBlock
import scipy.io as sio
from utils import get_downsamp_trans

def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        torch.nn.init.xavier_normal_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))

        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)

    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias.data)

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)


class Lipsync3DModel(nn.Module):
    def __init__(self, use_auto_regressive=False):
        super(Lipsync3DModel, self).__init__()
        self.use_auto_regressive=use_auto_regressive

        self.AudioEncoder = nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=72, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),   # 2 x 256 x 24 -> 72 x 128 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=72,out_channels=108, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 72 x 128 x 24 -> 108 x 64 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=108,out_channels=162, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 108 x 64 x 24 -> 162 x 32 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=162,out_channels=243, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 162 x 32 x 24 -> 243 x 16 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=243,out_channels=256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 243 x 16 x 24 -> 256 x 8 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256,out_channels=256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 256 x 8 x 24 -> 256 x 4 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256,out_channels=128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2), dilation=1),    # 256 x 4 x 24 -> 128 x 4 x 13
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128,out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2), dilation=1),    # 128 x 4 x 13 -> 64 x 4 x 8
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2), dilation=1),    # 64 x 4 x 8 -> 32 x 4 x 5
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2), dilation=1),    # 32 x 4 x 5 -> 16 x 4 x 4
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16,out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2), dilation=1),    # 16 x 4 x 4 -> 8 x 4 x 3
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8,out_channels=4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), dilation=1),     # 8 x 4 x 3 -> 4 x 4 x 2
            nn.LeakyReLU(),
            View([-1, 32]),
        )

        self.GeometryDecoder = nn.Sequential(
            nn.Linear(32, 150),
            nn.Dropout(0.5),
            nn.Linear(150, 1434)
        )

        self.TextureEncoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=1024, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024,out_channels=2048, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*4*2048, 2),
            nn.Tanh()
        )

        if self.use_auto_regressive:
            input_dim = 34
        else:
            input_dim = 32
        
        self.TextureDecoder = nn.Sequential(
            nn.Linear(input_dim, 4*4*1024),
            nn.ReLU(),
            View([-1, 1024, 4, 4]),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.Tanh(),
        )

    def forward(self, spec, texture_pred=None):
        # spec : B x 2 x 256 x 24
        # texture : B x 3 x 128 x 128

        latent = self.AudioEncoder(spec)
        geometry_diff = self.GeometryDecoder(latent)
        # if self.use_auto_regressive:
        #     texture_latent = self.TextureEncoder(texture_pred)
        #     latent = torch.cat([latent, texture_latent], 1)
        # texture_diff = self.TextureDecoder(latent)

        # return geometry_diff, texture_diff
        return geometry_diff


class landmark_to_BFM(nn.Module):
    def __init__(self):
        super(landmark_to_BFM, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(478*3, 478*3),
            nn.ReLU(),
            nn.Linear(478*3, 717),
            nn.ReLU(),
            nn.Linear(717, 358),
            nn.ReLU(),
            nn.Linear(358, 180),
            nn.ReLU(),
        )

        self.fc = nn.Linear(180, 64)
        self.tanh = nn.Tanh()
        self.apply(weights_init)
        torch.nn.init.zeros_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        out = 5 * self.tanh(self.fc(self.model(x)))

        return out


class MeshDecoderModel(nn.Module):
    def __init__(self, opt):
        super(MeshDecoderModel, self).__init__()
        self.device = opt.device
        self.refer_mesh = read_obj(os.path.join('renderer', 'data', 'bfm09_face_template.obj'))
        self.laplacians, self.downsamp_trans, self.upsamp_trans, self.pool_size = init_sampling(
        self.refer_mesh, os.path.join('renderer', 'data', 'params', 'bfm09_face'), 'bfm09_face')

        self.laplacians = [(torch.FloatTensor(np.array(laplacian.todense())) - torch.diag(torch.ones(laplacian.shape[0]))).to_sparse().to(self.device) for laplacian in self.laplacians]
        self.downsamp_trans = [torch.FloatTensor(np.array(downsamp_tran.todense())).to_sparse().to(self.device) for downsamp_tran in self.downsamp_trans]
        self.upsamp_trans = [torch.FloatTensor(np.array(upsamp_tran.todense())).to_sparse().to(self.device) for upsamp_tran in self.upsamp_trans]

        self.decoderF = [3, 16, 16, 3]
        self.K = 6

        self.decoder_fc = nn.Linear(478*3, self.pool_size[-2] * self.decoderF[0], bias=True)
        self.last_fc = nn.Linear(self.pool_size[-1] * self.decoderF[3], 64)
        self.relu = nn.ReLU()
        

        self.decoder_cheb_layers = nn.ModuleList([ChebResBlock(self.decoderF[0], self.decoderF[1], self.laplacians[-2], self.K),
                                                ChebResBlock(self.decoderF[1], self.decoderF[2], self.laplacians[-3], self.K),
                                                ChebResBlock(self.decoderF[2], self.decoderF[3], self.laplacians[-2], self.K)])

        self.init_fc_layers()

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

    
    def mesh_decoder(self, inputs):
        B = inputs.shape[0]
        layer1 = self.relu(self.decoder_fc(inputs))
        layer1 = layer1.reshape(B, self.pool_size[-2], self.decoderF[0])
        layer2 = self.decoder_cheb_layers[0](layer1)
        layer3 = self.poolwT(layer2, self.upsamp_trans[-2])
        layer3 = self.decoder_cheb_layers[1](layer3)
        layer4 = self.poolwT(layer3, self.downsamp_trans[-2])
        layer4 = self.decoder_cheb_layers[2](layer4)
        layer5 = self.poolwT(layer4, self.downsamp_trans[-1])
        layer5 = layer5.reshape(B, self.pool_size[-1] * self.decoderF[-1])
        outputs = self.last_fc(layer5)

        outputs = outputs.view(B, 64, 1)

        return outputs

    
    def init_fc_layers(self):
        nn.init.zeros_(self.last_fc.weight)
        nn.init.zeros_(self.last_fc.bias)

        nn.init.xavier_normal_(self.decoder_fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.decoder_fc.bias)

    def forward(self, mesh):
        outputs = self.mesh_decoder(mesh)

        return outputs


class Audio2GeometryModel(nn.Module):
    def __init__(self, opt, moodSize=12000, refer_idx=31):
        super(Audio2GeometryModel, self).__init__()
        self.device = opt.device
        self.refer_idx = refer_idx
        self.formantAnalysis = nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=72, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),   # 2 x 256 x 24 -> 72 x 128 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=72,out_channels=108, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 72 x 128 x 24 -> 108 x 64 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=108,out_channels=162, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 108 x 64 x 24 -> 162 x 32 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=162,out_channels=243, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 162 x 32 x 24 -> 243 x 16 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=243,out_channels=256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 243 x 16 x 24 -> 256 x 8 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256,out_channels=256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 256 x 8 x 24 -> 256 x 4 x 24
            nn.LeakyReLU(),
        )
        
        self.moodlen = 16
        mood = np.random.normal(.0, 0.01, (moodSize, self.moodlen))
        self.mood = nn.Parameter(
            torch.from_numpy(mood).float().view(moodSize, self.moodlen).to(self.device),
            requires_grad=True
        )
        
        self.articulation = nn.Sequential(
            nn.Conv2d(in_channels=256 + self.moodlen,out_channels=256 + self.moodlen, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2), dilation=1),    # (256+E) x 4 x 24 -> (256+E) x 4 x 13
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256 + self.moodlen,out_channels=256 + self.moodlen, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2), dilation=1),    # (256+E) x 4 x 13 -> (256+E) x 4 x 8
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256 + self.moodlen,out_channels=256 + self.moodlen, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2), dilation=1),    # (256+E) x 4 x 8 -> (256+E) x 4 x 5
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256 + self.moodlen,out_channels=256 + self.moodlen, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2), dilation=1),    # (256+E) x 4 x 5 -> (256+E) x 4 x 4
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256 + self.moodlen,out_channels=256 + self.moodlen, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), dilation=1),    # (256+E) x 4 x 4 -> (256+E) x 4 x 2
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256 + self.moodlen,out_channels=256 + self.moodlen, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), dilation=1),     # (256+E) x 4 x 2 -> (256+E) x 4 x 1
            nn.LeakyReLU(),
            View([-1, (256 + self.moodlen) * 4])
        )
        
        self.articulation_layer1 = 

        self.fc1 =  nn.Linear((256 + self.moodlen) * 4, 150)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(150, 442*3)
        
        self.init_fc_layers()        

    
    def init_fc_layers(self):
        # nn.init.zeros_(self.last_fc.weight)
        # nn.init.zeros_(self.last_fc.bias)

        # nn.init.xavier_normal_(self.decoder_fc.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.zeros_(self.decoder_fc.bias)

        nn.init.xavier_normal_(self.fc1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, spec, moodIdx=None):
        # input : spec( B x 2 x 256 x 24 )
        # output : geometry ( B x 478 x 3)
        B = spec.size()[0]
        if moodIdx==None:
            moodIdx = self.refer_idx * torch.ones((B), dtype=torch.long).to(self.device)
        formant_latent = self.formantAnalysis(spec)
        articulation_input = torch.cat((formant_latent,
                self.mood[moodIdx.chunk(chunks=1, dim=0)].view(B, self.moodlen, 1, 1).expand(B, self.moodlen, 4, 24)),
                dim=1        
        ).view(B, 256 + self.moodlen, 4, 24)
        audio_latent = self.articulation(articulation_input)
        exp_diff = self.fc2(self.tanh(self.fc1(audio_latent))).view(-1, 442, 3)

        return exp_diff

    
    