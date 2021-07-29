import math
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import os
from audiodvp_utils import util
from renderer.face_model import FaceModel
from facenet_pytorch import InceptionResnetV1


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
        self.pretrained_model = ResNet()
        self.fc = nn.Linear(2048, 257)  # 257 - 160 = 97
        self.face_model = FaceModel(opt=opt, batch_size=opt.batch_size)

        self.init_weights(opt.pretrained_model_path)

        # self.alpha = nn.Parameter(torch.zeros((1, 80, 1), device=opt.device))  # shared for all samples
        # self.beta = nn.Parameter(torch.zeros((1, 80, 1), device=opt.device))  # shared for all samples

        self.to(opt.device)

    def init_weights(self, pretrained_model_path):
        util.load_state_dict(self.pretrained_model, pretrained_model_path)

        torch.nn.init.zeros_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x, face_emb):
        coef = self.fc(self.pretrained_model(x)).unsqueeze(-1)

        alpha = coef[:, :80]
        beta = coef[:, 80:160]
        delta = coef[:, 160:224]
        rotation = coef[:, 251:254]
        translation = coef[:, 254:]
        gamma = coef[:, 224:251]

        render, mask, screen_vertices, tex, refined_tex = self.face_model(alpha, delta, beta, rotation, translation, gamma, face_emb=face_emb)

        return alpha, delta, beta, gamma, rotation, translation, render, mask, screen_vertices, tex, refined_tex


class CoefficientRegularization(nn.Module):
    def __init__(self):
        super(CoefficientRegularization, self).__init__()

    def forward(self, input):
        return torch.sum(input**2)


class PhotometricLoss(nn.Module):
    def __init__(self):
        super(PhotometricLoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, input, target):
        return self.mse(input, target)


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
    def __inint__(self):
        super(VarianceLoss, self).__init__()

    def forward(self, tex, skin_index):
        var_list = []
        for i in range(3):
            skin_tex = tex[:, skin_index, :]
            var_list.append(skin_tex[..., i].var(dim=1))
        
        loss = torch.cat(var_list).mean()

        return loss


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


class MouthMask(nn.Module):
    def __init__(self, opt):
        super(MouthMask, self).__init__()
        self.face_model = FaceModel(opt=opt, batch_size=1, load_model=True)
        self.tensor2pil = transforms.ToPILImage()

    def forward(self, alpha, delta, beta, gamma, rotation, translation, face_emb):
        delta = delta.clone()

        delta[0, 0, 0] = -8.0
        _, open_mask, _, _, _ = self.face_model(alpha, delta, beta, rotation, translation, gamma, face_emb, lower=True)

        delta[:, :, :] = 0.0
        _, close_mask, _, _, _ = self.face_model(alpha, delta, beta, rotation, translation, gamma, face_emb, lower=True)

        mouth_mask = torch.clamp(open_mask + close_mask, min=0.0, max=1.0)

        return mouth_mask

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