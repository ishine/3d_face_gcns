import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from models.networks.base_network import BaseNetwork
from models.networks.generator import AdaptiveFeatureGenerator, DomainClassifier, ReverseLayerF, ResidualBlock
from util.util import vgg_preprocess
import util.util as util
from .geomloss import SamplesLoss
from PIL import Image
import torch.nn.utils.spectral_norm as spectral_norm

class WTA_scale(torch.autograd.Function):
    """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """

    @staticmethod
    def forward(ctx, input, scale=1e-4):
        """
    In the forward pass we receive a Tensor containing the input and return a
    Tensor containing the output. You can cache arbitrary Tensors for use in the
    backward pass using the save_for_backward method.
    """
        activation_max, index_max = torch.max(input, -1, keepdim=True)
        input_scale = input * scale  # default: 1e-4
        # input_scale = input * scale  # default: 1e-4
        output_max_scale = torch.where(input == activation_max, input, input_scale)

        mask = (input == activation_max).type(torch.float)
        ctx.save_for_backward(input, mask)
        return output_max_scale

    @staticmethod
    def backward(ctx, grad_output):
        """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
        # import pdb
        # pdb.set_trace()
        input, mask = ctx.saved_tensors
        mask_ones = torch.ones_like(mask)
        mask_small_ones = torch.ones_like(mask) * 1e-4
        # mask_small_ones = torch.ones_like(mask) * 1e-4

        grad_scale = torch.where(mask == 1, mask_ones, mask_small_ones)
        grad_input = grad_output.clone() * grad_scale
        return grad_input, None

class VGG19_feature_color_torchversion(nn.Module):
    ''' 
    NOTE: there is no need to pre-process the input 
    input tensor should range in [0,1]
    '''

    def __init__(self, pool='max', vgg_normal_correct=False, ic=3):
        super(VGG19_feature_color_torchversion, self).__init__()
        self.vgg_normal_correct = vgg_normal_correct

        self.conv1_1 = nn.Conv2d(ic, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys, preprocess=True):
        ''' 
        NOTE: input tensor should range in [0,1]
        '''
        out = {}
        if preprocess:
            x = vgg_preprocess(x, vgg_normal_correct=self.vgg_normal_correct)
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]

class NoVGGCorrespondence(BaseNetwork):
    # input is Al, Bl, channel = 1, range~[0,255]
    def __init__(self, opt):
        self.opt = opt
        super().__init__()

        # self.p, self.blur = 1, opt.blur
        self.p, self.blur = 1, 0.025#0.005
        self.uot = SamplesLoss("sinkhorn", p=self.p, blur=self.blur,
                                      debias=False, potentials=True)

        self.adaptive_model_seg = AdaptiveFeatureGenerator(opt, opt.semantic_nc)
        self.adaptive_model_img = AdaptiveFeatureGenerator(opt, opt.ref_nc)
        if opt.weight_domainC > 0 and (not opt.domain_rela):
            self.domain_classifier = DomainClassifier(opt)

        self.down = opt.warp_stride # 4

        self.feat_ch = 64
        self.cor_dim = 256
        coord_c = 3 if opt.use_coordconv else 0

        self.layer_ref = nn.Sequential(
            ResidualBlock(self.feat_ch * 4 + coord_c, self.feat_ch * 4 + coord_c, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feat_ch * 4 + coord_c, self.feat_ch * 4 + coord_c, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feat_ch * 4 + coord_c, self.feat_ch * 4 + coord_c, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feat_ch * 4 + coord_c, self.feat_ch * 4 + coord_c, kernel_size=3, padding=1, stride=1))
        
        self.layer_cont = nn.Sequential(
            ResidualBlock(self.feat_ch * 4 + coord_c, self.feat_ch * 4 + coord_c, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feat_ch * 4 + coord_c, self.feat_ch * 4 + coord_c, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feat_ch * 4 + coord_c, self.feat_ch * 4 + coord_c, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feat_ch * 4 + coord_c, self.feat_ch * 4 + coord_c, kernel_size=3, padding=1, stride=1))

        self.phi = nn.Conv2d(in_channels=self.feat_ch * 4 + coord_c, out_channels=self.cor_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=self.feat_ch * 4 + coord_c, out_channels=self.cor_dim, kernel_size=1, stride=1, padding=0)

        self.phi_w = nn.Conv2d(in_channels=self.feat_ch * 4 + coord_c, out_channels=self.cor_dim, kernel_size=1, stride=1, padding=0)
        self.theta_w = nn.Conv2d(in_channels=self.feat_ch * 4 + coord_c, out_channels=self.cor_dim, kernel_size=1, stride=1, padding=0)

        self.phi_conf = nn.Conv2d(in_channels=self.feat_ch * 4 + coord_c, out_channels=self.cor_dim, kernel_size=1, stride=1, padding=0)
        self.theta_conf = nn.Conv2d(in_channels=self.feat_ch * 4 + coord_c, out_channels=self.cor_dim, kernel_size=1, stride=1, padding=0)
        # self.theta_atten = nn.Conv2d(in_channels=self.feat_ch * 4 + label_nc, out_channels=self.cor_dim, kernel_size=1, stride=1, padding=0, bias=False)

        # self.upsampling_bi = F.interpolate(scale_factor=self.down, mode='bilinear')
        if opt.warp_bilinear:
            self.upsampling = nn.Upsample(scale_factor=self.down, mode='bilinear')
        else:
            self.upsampling = nn.Upsample(scale_factor=self.down)
        self.zero_tensor = None
        self.relu = nn.ReLU()

    def forward(self, ref_img, real_img, seg_map, detach_flag=False):
        coor_out = {}
        batch_size, _, im_height, im_width = ref_img.shape
        feat_height, feat_width = int(im_height / self.down), int(im_width / self.down)

        seg_feat2, seg_feat3, seg_feat4, seg_feat5, seg_feat6 = self.adaptive_model_seg(seg_map, seg_map, multi=True)
        ref_feat2, ref_feat3, ref_feat4, ref_feat5, ref_feat6 = self.adaptive_model_img(ref_img, ref_img, multi=True)

        adp_feat_seg = util.feature_normalize(seg_feat6)
        adp_feat_img = util.feature_normalize(ref_feat6)

        # if self.opt.isTrain and self.opt.novgg_featpair > 0:
        #     adaptive_feature_img_pair = self.adaptive_model_img(real_img, real_img)
        #     adaptive_feature_img_pair = util.feature_normalize(adaptive_feature_img_pair)
        #     coor_out['loss_novgg_featpair'] = F.l1_loss(adp_feat_seg, adaptive_feature_img_pair) * self.opt.novgg_featpair

        if self.opt.use_coordconv:
            adp_feat_seg = self.addcoords(adp_feat_seg)
            adp_feat_img = self.addcoords(adp_feat_img)

        cont_features = self.layer_cont(adp_feat_seg)
        ref_features = self.layer_ref(adp_feat_img)

        dim_mean = 1 if self.opt.PONO_C else -1

        # feature branch
        theta, phi = cont_features, ref_features
        theta = self.theta(theta)
        theta = F.unfold(theta, kernel_size=self.opt.match_kernel, padding=int(self.opt.match_kernel // 2))
        theta = util.mean_normalize(theta, dim_mean=dim_mean)
        theta_permute = theta.permute(0, 2, 1).contiguous()

        phi = self.phi(phi)
        phi = F.unfold(phi, kernel_size=self.opt.match_kernel, padding=int(self.opt.match_kernel // 2))
        phi = util.mean_normalize(phi, dim_mean=dim_mean)
        phi_permute = phi.permute(0, 2, 1).contiguous()


        # weight branch
        phi_w_feat = self.phi_w(ref_features).view(batch_size, self.cor_dim, -1)
        phi_w_feat = util.mean_normalize(phi_w_feat, dim_mean=dim_mean)
        phi_w_feat_pmt = phi_w_feat.permute(0, 2, 1).contiguous()

        theta_w_feat = self.theta_w(cont_features).view(batch_size, self.cor_dim, -1)
        theta_w_feat = util.mean_normalize(theta_w_feat, dim_mean=dim_mean)
        theta_w_feat_pmt = theta_w_feat.permute(0, 2, 1).contiguous()

        _, N, D = theta_permute.shape
        theta_w = torch.matmul(theta_w_feat_pmt, phi_w_feat.mean(-1, keepdim=True))
        theta_w = F.softmax(theta_w.view(batch_size, N), dim=-1)
        phi_w = torch.matmul(phi_w_feat_pmt, theta_w_feat.mean(-1, keepdim=True))
        phi_w = F.softmax(phi_w.view(batch_size, N), dim=-1)
        theta_w_ = theta_w.view(-1, 1, 64, 64).repeat(1, 3, 1, 1)
        phi_w_ = phi_w.view(-1, 1, 64, 64).repeat(1, 3, 1, 1)
        coor_out['weight1'] = F.interpolate(theta_w_, size=(256, 256)).detach()
        coor_out['weight2'] = F.interpolate(phi_w_, size=(256, 256)).detach()


        # confidence branch
        phi_conf_feat = self.phi_conf(ref_features).view(batch_size, self.cor_dim, -1)
        phi_conf_feat = util.mean_normalize(phi_conf_feat, dim_mean=dim_mean)
        # phi_conf_feat_pmt = phi_conf_feat.permute(0, 2, 1).contiguous()
        theta_conf_feat = self.theta_conf(cont_features).view(batch_size, self.cor_dim, -1)
        theta_conf_feat = util.mean_normalize(theta_conf_feat, dim_mean=dim_mean)
        theta_conf_feat_pmt = theta_conf_feat.permute(0, 2, 1).contiguous()
        conf_map = torch.matmul(theta_conf_feat_pmt, phi_conf_feat)
        conf_map = torch.max(conf_map, -1, keepdim=True)[0]
        conf_map = (conf_map - conf_map.mean(dim=1, keepdim=True)).view(batch_size, 1, feat_height, feat_width)
        conf_map = torch.sigmoid(conf_map*10.0)

        conf_map_ = conf_map.view(-1, 1, 64, 64).repeat(1, 3, 1, 1)
        coor_out['conf_map'] = F.interpolate(conf_map_, size=(256, 256))

        # OT matching branch
        F_, G_ = self.uot(theta_w, theta_permute, phi_w, phi_permute)
        F_i, G_j = F_.view(-1, N, 1), G_.view(-1, 1, N)
        a_i, b_j = theta_w.view(-1, N, 1), phi_w.view(-1, N, 1)
        C_ij = 1 - torch.matmul(theta_permute, phi)
        eps = self.blur ** self.p
        f = ((F_i + G_j - C_ij) / eps).exp() * (a_i * b_j)
        f_div_C = f / f.sum(-1).view(-1, N, 1)

        # feature transport branch
        ref_ = F.interpolate(ref_img, size=(64, 64), mode='nearest')
        channel_ = ref_.shape[1]
        ref_ = ref_.view(batch_size, channel_, -1)
        ref_ = ref_.permute(0, 2, 1)
        y1 = torch.matmul(f_div_C, ref_)

        y_ = y1.permute(0, 2, 1).contiguous()
        y_ = y_.view(batch_size, channel_, feat_height, feat_width)  # 2*3*44*44
        coor_out['warp_tmp'] = self.upsampling(y_)

        ref_feat2 = F.interpolate(ref_feat2, size=(64, 64), mode='nearest')
        channel2 = ref_feat2.shape[1]
        ref_feat2 = ref_feat2.view(batch_size, channel2, -1).permute(0, 2, 1)
        y2 = torch.matmul(f_div_C, ref_feat2)

        ref_feat3 = F.interpolate(ref_feat3, size=(64, 64), mode='nearest')
        channel3 = ref_feat3.shape[1]
        ref_feat3 = ref_feat3.view(batch_size, channel3, -1).permute(0, 2, 1)
        y3 = torch.matmul(f_div_C, ref_feat3)

        ref_feat4 = F.interpolate(ref_feat4, size=(64, 64), mode='nearest')
        channel4 = ref_feat4.shape[1]
        ref_feat4 = ref_feat4.view(batch_size, channel4, -1).permute(0, 2, 1)
        y4 = torch.matmul(f_div_C, ref_feat4)

        ref_feat5 = F.interpolate(ref_feat5, size=(64, 64), mode='nearest')
        channel5 = ref_feat5.shape[1]
        ref_feat5 = ref_feat5.view(batch_size, channel5, -1).permute(0, 2, 1)
        y5 = torch.matmul(f_div_C, ref_feat5)

        y1 = y1.permute(0, 2, 1).view(batch_size, channel_, feat_height, feat_width)
        y2 = y2.permute(0, 2, 1).view(batch_size, channel2, feat_height, feat_width)
        y3 = y3.permute(0, 2, 1).view(batch_size, channel3, feat_height, feat_width)
        y4 = y4.permute(0, 2, 1).view(batch_size, channel4, feat_height, feat_width)
        y5 = y5.permute(0, 2, 1).view(batch_size, channel5, feat_height, feat_width)

        coor_out['warp_out'] = [seg_map, seg_feat2, seg_feat3, seg_feat4, seg_feat5, y1, y2, y3, y4, y5, conf_map]

        if self.opt.warp_cycle_w > 0:
            f_WTA_v = f.transpose(1, 2)
            f_div_C_v = f_WTA_v / f_WTA_v.sum(-1).view(-1, N, 1)

            channel = y_.shape[1]
            y_ = y_.view(batch_size, channel, -1).permute(0, 2, 1)
            warp_cycle = torch.matmul(f_div_C_v, y_).permute(0, 2, 1).contiguous()

            coor_out['warp_cycle'] = warp_cycle.view(batch_size, channel, feat_height, feat_width)

        return coor_out

    def addcoords(self, x):
        bs, _, h, w = x.shape
        xx_ones = torch.ones([bs, h, 1], dtype=x.dtype, device=x.device)
        xx_range = torch.arange(w, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(1)
        xx_channel = torch.matmul(xx_ones, xx_range).unsqueeze(1)

        yy_ones = torch.ones([bs, 1, w], dtype=x.dtype, device=x.device)
        yy_range = torch.arange(h, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(-1)
        yy_channel = torch.matmul(yy_range, yy_ones).unsqueeze(1)

        xx_channel = xx_channel.float() / (w - 1)
        yy_channel = yy_channel.float() / (h - 1)
        xx_channel = 2 * xx_channel - 1
        yy_channel = 2 * yy_channel - 1

        rr_channel = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))

        concat = torch.cat((x, xx_channel, yy_channel, rr_channel), dim=1)
        return concat
