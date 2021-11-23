import sys
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')
import math
import torch
import torch.nn as nn
from torchvision import transforms
from audiodvp_utils import util
from renderer.face_model import FaceModel
from .conv import Conv2d
from renderer.mesh_refiner import ChebResBlock, ChebConv
from lib.mesh_io import read_obj
import os
from gcn_util import utils
import numpy as np
import scipy.io as sio
from torch.nn import functional as F
from lipsync3d.utils import get_downsamp_trans


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


class ResnetFaceModelOptimizer(nn.Module):
    def __init__(self, opt):
        super(ResnetFaceModelOptimizer, self).__init__()
        self.opt = opt
        self.pretrained_model = ResNet()
        self.fc = nn.Linear(2048, 97)  # 257 - 160 = 97
        self.face_model = FaceModel(opt=opt, batch_size=opt.batch_size)
        self.alpha_resnet = ResNet()
        self.beta_resnet = ResNet()
        self.alpha_fc = nn.Linear(2048, 80)
        self.beta_fc = nn.Linear(2048, 80)
        
        self.init_weights(opt.pretrained_model_path)

        # self.alpha = nn.Parameter(torch.zeros((1, 80, 1), device=opt.device))  # shared for all samples
        # self.beta = nn.Parameter(torch.zeros((1, 80, 1), device=opt.device))  # shared for all samples
        self.alpha_beta_input = torch.zeros((1, 3, 224, 224), device=opt.device)

        self.to(opt.device)

    def init_weights(self, pretrained_model_path):
        util.load_state_dict(self.pretrained_model, pretrained_model_path)
        util.load_state_dict(self.alpha_resnet, pretrained_model_path)
        util.load_state_dict(self.beta_resnet, pretrained_model_path)

        torch.nn.init.zeros_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)
        torch.nn.init.zeros_(self.alpha_fc.weight)
        torch.nn.init.zeros_(self.alpha_fc.bias)
        torch.nn.init.zeros_(self.beta_fc.weight)
        torch.nn.init.zeros_(self.beta_fc.bias)

    def forward(self, x, use_refine=True):
        coef = self.fc(self.pretrained_model(x)).unsqueeze(-1)
        alpha = self.alpha_fc(self.alpha_resnet(self.alpha_beta_input)).unsqueeze(-1)
        beta = self.beta_fc(self.beta_resnet(self.alpha_beta_input)).unsqueeze(-1)
        
        delta = coef[:, 0:64]
        rotation = coef[:, 91:94]
        translation = coef[:, 94:]
        gamma = coef[:, 64:91]

        render, mask, screen_vertices, tex, refined_tex = self.face_model(alpha, delta, beta, rotation, translation, gamma, use_refine=use_refine)

        return alpha, delta, beta, gamma, rotation, translation, render, mask, screen_vertices, tex, refined_tex
    

class ResnetDeltaModel(nn.Module):
    def __init__(self, opt):
        super(ResnetDeltaModel, self).__init__()
        self.pretrained_model = ResNet()
        self.fc = nn.Linear(2048, 64)
        self.facemodel = FaceModel(opt=opt, batch_size=opt.batch_size)

        self.init_weights(opt.pretrained_model_path)
        self.to(opt.device)
        
        self.downsamp_trans = get_downsamp_trans()
        mat_data = sio.loadmat(opt.matlab_data_path)
        exp_base = torch.from_numpy(mat_data['exp_base']).reshape(-1, 3 * 64)
        self.exp_base = torch.mm(self.downsamp_trans, exp_base).reshape(-1, 64).unsqueeze(0).expand(opt.batch_size, -1, -1).to(opt.device)

    def init_weights(self, pretrained_model_path):
        util.load_state_dict(self.pretrained_model, pretrained_model_path)

        torch.nn.init.zeros_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x, reference_geometry, rotation, translation):
        delta = self.fc(self.pretrained_model(x)).unsqueeze(-1) # B x 64 x 1
        geometry_pred = reference_geometry + self.exp_base.bmm(delta).view(-1, 478, 3)
        
        mediapipe_mesh_pred = self.facemodel.geometry_to_pixel(geometry_pred, rotation, translation)
        return delta, mediapipe_mesh_pred


class CoefficientRegularization(nn.Module):
    def __init__(self, use_mean=False):
        super(CoefficientRegularization, self).__init__()
        self.use_mean = use_mean
    def forward(self, input):
        if self.use_mean:
            output = torch.sum(torch.mean(input**2, dim=0))
        else:
            output = torch.sum(input**2)
        return output


class PhotometricLoss(nn.Module):
    def __init__(self):
        super(PhotometricLoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, input, target):
        return self.mse(input, target)


class LandmarkLoss(nn.Module):
    def __init__(self, opt, use_mean=False):
        super(LandmarkLoss, self).__init__()

        self.device = opt.device
        self.use_mean = use_mean
        self.landmark_weight = torch.tensor([[
                1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                1.0,  1.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
                50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
        ]], device=self.device).unsqueeze(-1)

    def forward(self, landmark, landmark_gt):
        landmark_loss = (landmark - landmark_gt) ** 2
        
        if self.use_mean:
            landmark_loss = torch.mean(self.landmark_weight * landmark_loss)
        else:
            landmark_loss = torch.sum(self.landmark_weight * landmark_loss) / 68.0

        return landmark_loss

class VertexWiseLoss(nn.Module):
    def __init__(self):
        super(VertexWiseLoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, tex, refined_tex):
        return self.mse(tex, refined_tex)

class IdentityPreservingLoss(nn.Module):
    def __init__(self):
        super(IdentityPreservingLoss, self).__init__()
        self.cosine = torch.nn.CosineSimilarity()

    def forward(self, face_emb, render_emb):
        return (1 - self.cosine(face_emb, render_emb)).mean()


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
    
    def forward(self, prediction, target_is_real):
        if target_is_real:
            loss = -prediction.mean()
        else:
            loss = prediction.mean()
        
        return loss

class VarianceLoss(nn.Module):
    def __init__(self):
        super(VarianceLoss, self).__init__()

    def forward(self, tex, skin_index):
        var_list = []
        for i in range(3):
            skin_tex = tex[:, skin_index, :]
            var_list.append(skin_tex[..., i].var(dim=1))
        
        loss = torch.cat(var_list).mean()

        return loss


class SyncLoss(nn.Module):
    def __init__(self, device):
        super(SyncLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.device = device

    def forward(self, audio_emb, coef_emb):
        cosine_sim = nn.functional.cosine_similarity(audio_emb, coef_emb)
        label = torch.ones(cosine_sim.size(0), 1).float().to(self.device)
        loss = self.bce(cosine_sim.unsqueeze(1), label)

        return loss


class WeightedMSELoss(nn.Module):
    def __init__(self, weight):
        super(WeightedMSELoss, self).__init__()
        self.weight = weight
    
    def forward(self, input, target):
        
        return (self.weight * (input - target) ** 2).mean()


class WeightedL1Loss(nn.Module):
    def __init__(self, weight):
        super(WeightedL1Loss, self).__init__()
        self.weight = weight
    
    def forward(self, input, target):
        
        return (self.weight * torch.abs(input - target)).mean()


class WeightedRMSLoss(nn.Module):
    def __init__(self, weight):
        super(WeightedRMSLoss, self).__init__()
        self.weight = weight
    
    def forward(self, input, target):
        return torch.sqrt((self.weight * (input - target)**2).mean())


class MaskedVertexRMSLoss(nn.Module):
    def __init__(self, exp_base, mask):
        super(MaskedVertexRMSLoss, self).__init__()
        self.exp_base = exp_base
        self.mask = mask

    def forward(self, input, target):

        B = input.shape[0]
        diff_expression = input - target
        diff_vertices = self.exp_base.matmul(diff_expression).reshape(B, -1, 3)
        return torch.sqrt(torch.mean(self.mask * (diff_vertices ** 2)))


class AudioExpressionModule(nn.Module):
    def __init__(self, opt):
        super(AudioExpressionModule, self).__init__()
        self.opt = opt
        self.conv1 = nn.Conv1d(opt.Nw, 5, 3)
        self.conv2 = nn.Conv1d(5, 3, 3)
        self.conv3 = nn.Conv1d(3, 1, 3)
        self.fc = nn.Linear(250, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.unsqueeze(-1)


class ExpressionEstimator_Attention(nn.Module):
    def __init__(self, seq_len):
        super(ExpressionEstimator_Attention, self).__init__()
        self.seq_len = seq_len
        self.subspace_dim = 32

        self.convNet = nn.Sequential(
            nn.Conv2d(29, 32, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), #  29 x 16 x 1 => 32 x 8 x 1
            nn.LeakyReLU(0.02, True),
            nn.Conv2d(32, 32, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 32 x 8 x 1 => 32 x 4 x 1
            nn.LeakyReLU(0.02, True),
            nn.Conv2d(32, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 32 x 4 x 1 => 64 x 2 x 1
            nn.LeakyReLU(0.02, True),
            nn.Conv2d(64, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 64 x 2 x 1 => 64 x 1 x 1
            nn.LeakyReLU(0.02, True),
        )

        self.fullNet = nn.Sequential(
            nn.Linear(in_features = 64, out_features=128, bias = True),
            nn.LeakyReLU(0.02),
            nn.Linear(in_features = 128, out_features=64, bias = True),
            nn.LeakyReLU(0.02),
            nn.Linear(in_features=64, out_features=self.subspace_dim, bias = True),
            nn.Tanh()
        )

        self.last_fc = nn.Linear(in_features=self.subspace_dim, out_features=64, bias = False)

        # attention
        self.attentionConvNet = nn.Sequential( # b x subspace_dim x seq_len
            nn.Conv1d(self.subspace_dim, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features = self.seq_len, out_features=self.seq_len, bias = True),   
            nn.Softmax(dim=1)
            )

        self.apply(weights_init)
        torch.nn.init.normal_(self.last_fc.weight, 0.0, 0.02)

            
    def forward(self, audio_features):
        result_subspace, intermediate_expression = self.getAudioExpressions(audio_features)
        b = result_subspace.shape[0]
        result = 10.0 * self.last_fc(result_subspace.reshape(b, 1, self.subspace_dim)).reshape(b, 64, 1)
        result_intermediate = 10.0 * self.last_fc(intermediate_expression.reshape(b, 1, self.subspace_dim)).reshape(b, 64, 1)
        return result, result_intermediate

    def getAudioExpressions(self, audio_features):
        # audio_features: b x seq_len x 16 x 29
        b = audio_features.shape[0] # batchsize
        audio_features = audio_features.reshape(b * self.seq_len, 1, 16, 29) # b * seq_len x 1 x 16 x 29
        audio_features = torch.transpose(audio_features, 1, 3).contiguous() # b* seq_len  x 29 x 16 x 1
        conv_res = self.convNet(audio_features) # b*seq_len x 64 x 1 x 1
        conv_res = torch.reshape(conv_res, (b * self.seq_len, 1, -1))   # b*seq_len x 1 x 64
        result_subspace = self.fullNet(conv_res)[:,0,:] # b * seq_len x subspace_dim
        result_subspace = result_subspace.reshape(b, self.seq_len, self.subspace_dim)# b x seq_len x subspace_dim

        #################
        ### attention ###
        ################# 
        result_subspace_T = torch.transpose(result_subspace, 1, 2) # b x subspace_dim x seq_len
        intermediate_expression = result_subspace_T[:,:,(self.seq_len // 2):(self.seq_len // 2) + 1] # b x subspace_dim x 1
        att_conv_res = self.attentionConvNet(result_subspace_T)
        #print('att_conv_res', att_conv_res.shape)
        attention = self.attentionNet(att_conv_res.reshape(b, self.seq_len)).reshape(b, self.seq_len, 1) # b x seq_len x 1
        #print('attention', attention.shape)
        # pooling along the sequence dimension
        result_subspace = torch.bmm(result_subspace_T, attention)
        #print('result_subspace', result_subspace.shape)
        ###

        return result_subspace.reshape(b, self.subspace_dim, 1), intermediate_expression


class MouthMask(nn.Module):
    def __init__(self, opt):
        super(MouthMask, self).__init__()
        self.face_model = FaceModel(opt=opt, batch_size=1, load_model=True)
        self.tensor2pil = transforms.ToPILImage()

    def forward(self, alpha, delta, beta, gamma, rotation, translation):
        delta = delta.clone()

        delta[0, 0, 0] = -8.0
        _, open_mask, _, _, _ = self.face_model(alpha, delta, beta, rotation, translation, gamma, lower=True)

        delta[:, :, :] = 0.0
        _, close_mask, _, _, _ = self.face_model(alpha, delta, beta, rotation, translation, gamma, lower=True)

        mouth_mask = torch.clamp(open_mask + close_mask, min=0.0, max=1.0)

        return mouth_mask

def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        torch.nn.init.xavier_normal_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))

        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)

    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias.data)


class Wav2Delta(nn.Module):
    def __init__(self, opt, output_dim=64):
        super(Wav2Delta, self).__init__()
        self.device = opt.device
        self.exp_base, self.trans_mat = self.get_exp_matrices()

        self.refer_mesh = read_obj(os.path.join('renderer', 'data', 'bfm09_face_template.obj'))
        self.laplacians, self.downsamp_trans, self.upsamp_trans, self.pool_size = utils.init_sampling(
        self.refer_mesh, os.path.join('renderer', 'data', 'params', 'bfm09_face'), 'bfm09_face')
        self.laplacians = [(torch.FloatTensor(np.array(laplacian.todense())) - torch.diag(torch.ones(laplacian.shape[0]))).to_sparse().to(self.device) for laplacian in self.laplacians]
        self.upsamp_trans = [torch.FloatTensor(np.array(upsamp_tran.todense())).to_sparse().to(self.device) for upsamp_tran in self.upsamp_trans]
        self.decoderF = [32, 16, 16, 16, 3]
        self.K = 6

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

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)
            
        self.decoder_fc = nn.Linear(512, self.pool_size[-1] * self.decoderF[0], bias=True)
        self.relu = nn.ReLU()

        self.decoder_cheb_layers = nn.ModuleList([ChebResBlock(self.decoderF[0], self.decoderF[1], self.laplacians[-2], self.K),
                                                ChebResBlock(self.decoderF[1], self.decoderF[2], self.laplacians[-3], self.K),
                                                ChebResBlock(self.decoderF[2], self.decoderF[3], self.laplacians[-4], self.K),
                                                ChebResBlock(self.decoderF[3], self.decoderF[4], self.laplacians[-5], self.K)])

        self.last_cheb = ChebConv(3, 3, self.laplacians[0], self.K, zero_init=True)

        self.fc = nn.Sequential(nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU())
        self.last_fc = nn.Linear(128, output_dim)

        pretrained_audio_encoder_path = "models/wav2delta_pretrained.pth"
        self.init_weights(pretrained_audio_encoder_path)

    
    def init_weights(self, pretrained_model_path):
        ckpt = torch.load(pretrained_model_path, map_location='cpu')

        own_state = self.state_dict()
        pretrained_dict = {k: v for k, v in ckpt.items() if k in own_state}
        own_state.update(pretrained_dict)

        self.load_state_dict(own_state)

        for p in self.audio_encoder.parameters():
            p.requires_grad = False

        nn.init.xavier_normal_(self.decoder_fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.decoder_fc.bias)

        self.fc.apply(weights_init)
        nn.init.zeros_(self.last_fc.weight)
        nn.init.zeros_(self.last_fc.bias)


    def get_exp_matrices(self):
        mat_data = sio.loadmat('renderer/data/data.mat')
        exp_base = torch.from_numpy(mat_data['exp_base']).to(self.device).double()
        exp_base_T = exp_base.T
        inv_sym_base = exp_base_T.matmul(exp_base).inverse()
        trans_mat = inv_sym_base.matmul(exp_base.T).to(self.device)

        return exp_base, trans_mat

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

    def forward(self, x):
        # x = (B, T, 1, 80, 16)
        B = x.size(0)
        input_dim_size = len(x.size())
        if input_dim_size > 4:
            x = torch.cat([x[:, i] for i in range(x.size(1))], dim=0)

        out = self.audio_encoder(x).flatten(start_dim=1)
        # out = F.normalize(out, p=2, dim=1)
        # newB = out.shape[0]
        # layer1 = self.relu(self.decoder_fc(out))
        # layer1 = layer1.reshape(newB, self.pool_size[-1], self.decoderF[0])
        # layer2 = self.poolwT(layer1, self.upsamp_trans[-1])
        # layer2 = self.decoder_cheb_layers[0](layer2)
        # layer3 = self.poolwT(layer2, self.upsamp_trans[-2])
        # layer3 = self.decoder_cheb_layers[1](layer3)
        # layer4 = self.poolwT(layer3, self.upsamp_trans[-3])
        # layer4 = self.decoder_cheb_layers[2](layer4)
        # layer5 = self.poolwT(layer4, self.upsamp_trans[-4])
        # out = self.decoder_cheb_layers[3](layer5)
        # out = self.last_cheb(out)
        # out = out.reshape(newB, -1, 1).double()
        # out = self.trans_mat.matmul(out).squeeze().float()
        out = self.last_fc(self.fc(out))


        if input_dim_size > 4:
            out = torch.split(out, B, dim=0)  # [(B, 64) * T]
            out = torch.stack(out, dim=1)
        return out


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None