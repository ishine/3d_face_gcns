import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
import scipy.sparse as sp
from lib.mesh_io import read_obj
import os
import utils
from lib import graph

class ChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, laplacian, K, is_last=False):
        super(ChebConv, self).__init__()

        self.laplacian = laplacian
        self.fc = nn.Linear(K * in_channels, out_channels)
        self.K = K
        
        if is_last:
            nn.init.xavier_normal_(self.fc.weight, gain=nn.init.calculate_gain('tanh'))
        else:
            nn.init.xavier_normal_(self.fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        out_list = []
        B, N, Fin = x.shape
        B, N, Fin = int(B), int(N), int(Fin)
        x0 = x.permute(1, 0, 2)
        x0 = x0.reshape(N, B*Fin)
        Tx_0 = x0 # N x B*Fin
        out_list.append(Tx_0) # N x B*Fin

        if self.K > 1:
            Tx_1 = torch.mm(self.laplacian, x0)
            out_list.append(Tx_1)
        
        for k in range(2, self.K):
            Tx_2 = 2 * torch.mm(self.laplacian, Tx_1) - Tx_0
            out_list.append(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2
        
        out_list = [out.reshape(N, B, Fin) for out in out_list]
        out = torch.cat(out_list, dim=2)
        out = out.reshape(N*B, self.K * Fin)
        out = self.fc(out)
        
        out = out.reshape(N, B, -1)
        out = out.permute(1, 0, 2)

        return out


class ChebResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, laplacian, K):
        super(ChebResBlock, self).__init__()
        if in_channels != out_channels:
            self.shortcut = ChebConv(in_channels, out_channels, laplacian, K)
        else:
            self.shortcut = None
        
        self.cheby1 = ChebConv(in_channels, out_channels, laplacian, K)
        self.cheby2 = ChebConv(out_channels, out_channels, laplacian, K)
        self.relu = nn.ReLU()


    def forward(self, input):
        if self.shortcut == None:
            shortcut = input
        else:
            shortcut = self.shortcut(input)

        x = self.relu(self.cheby1(input))
        x = self.relu(self.cheby2(x) + shortcut)

        return x


class MeshRefinementModel(nn.Module):
    def __init__(self):
        super(MeshRefinementModel, self).__init__()
        self.device = torch.device('cuda')
        self.refer_mesh = read_obj(os.path.join('renderer', 'data', 'bfm09_face_template.obj'))
        self.laplacians, self.downsamp_trans, self.upsamp_trans, self.pool_size = utils.init_sampling(
        self.refer_mesh, os.path.join('renderer', 'data', 'params', 'bfm09_face'), 'bfm09_face')
        self.laplacians = [(torch.FloatTensor(np.array(laplacian.todense())) - torch.diag(torch.ones(laplacian.shape[0]))).to_sparse().to(self.device) for laplacian in self.laplacians]
        self.downsamp_trans = [torch.FloatTensor(np.array(downsamp_tran.todense())).to_sparse().to(self.device) for downsamp_tran in self.downsamp_trans]
        self.upsamp_trans = [torch.FloatTensor(np.array(upsamp_tran.todense())).to_sparse().to(self.device) for upsamp_tran in self.upsamp_trans]

        self.decoderF = [32, 16, 16, 16, 3]
        self.refineF = [3, 16, 32, 32, 16, 3]
        self.K = 6

        self.decoder_fc = nn.Linear(512, self.pool_size[-1] * self.decoderF[0], bias=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        nn.init.xavier_normal_(self.decoder_fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.decoder_fc.bias)

        self.decoder_cheb_layers = nn.ModuleList([ChebResBlock(self.decoderF[0], self.decoderF[1], self.laplacians[-2], self.K),
                                                ChebResBlock(self.decoderF[1], self.decoderF[2], self.laplacians[-3], self.K),
                                                ChebResBlock(self.decoderF[2], self.decoderF[3], self.laplacians[-4], self.K),
                                                ChebResBlock(self.decoderF[3], self.decoderF[4], self.laplacians[-5], self.K)])

        self.refiner_cheb_layers = nn.ModuleList([ChebResBlock(self.refineF[0], self.refineF[1], self.laplacians[0], self.K),
                                                ChebResBlock(self.refineF[1], self.refineF[2], self.laplacians[1], self.K),
                                                ChebResBlock(self.refineF[2], self.refineF[3], self.laplacians[1], self.K),
                                                ChebResBlock(self.refineF[3], self.refineF[4], self.laplacians[0], self.K),
                                                ChebResBlock(self.refineF[4], self.refineF[5], self.laplacians[0], self.K)])

        self.combiner = ChebConv(6, 3, self.laplacians[0], self.K, is_last=True)

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
        layer1 = layer1.reshape(B, self.pool_size[-1], self.decoderF[0])
        layer2 = self.poolwT(layer1, self.upsamp_trans[-1])
        layer2 = self.decoder_cheb_layers[0](layer2)
        layer3 = self.poolwT(layer2, self.upsamp_trans[-2])
        layer3 = self.decoder_cheb_layers[1](layer3)
        layer4 = self.poolwT(layer3, self.upsamp_trans[-3])
        layer4 = self.decoder_cheb_layers[2](layer4)
        layer5 = self.poolwT(layer4, self.upsamp_trans[-4])
        outputs = self.decoder_cheb_layers[3](layer5)

        return outputs


    def mesh_refiner(self, inputs):
        layer1 = self.refiner_cheb_layers[0](inputs)
        layer2 = self.poolwT(layer1, self.downsamp_trans[0])
        layer2 = self.refiner_cheb_layers[1](layer2)
        layer3 = self.refiner_cheb_layers[2](layer2)
        layer4 = self.poolwT(layer3, self.upsamp_trans[0])
        layer4 = self.refiner_cheb_layers[3](layer4)
        outputs = self.refiner_cheb_layers[4](layer4)

        return outputs


    def forward(self, image_emb, tex):
        decode_color = self.mesh_decoder(image_emb)
        refine_color = self.mesh_refiner(tex)
        concat = torch.cat([decode_color, refine_color], dim = -1)
        outputs = self.combiner(concat)
        outputs = self.tanh(outputs)
        
        return outputs