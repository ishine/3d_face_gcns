import torch
import torch.nn.functional as F
import models.networks as networks
import util.util as util


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.net = torch.nn.ModuleDict(self.initialize_networks(opt))

        # set loss functions
        if opt.isTrain:
            self.vggnet_fix = networks.correspondence.VGG19_feature_color_torchversion(vgg_normal_correct=opt.vgg_normal_correct)
            self.vggnet_fix.load_state_dict(torch.load('vendor/UNITE/models/vgg19_conv.pth'))
            self.vggnet_fix.eval()
            for param in self.vggnet_fix.parameters():
                param.requires_grad = False

            self.vggnet_fix.to(opt.gpu_ids[0])
            self.contextual_forward_loss = networks.ContextualLoss_forward(opt)

            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.MSE_loss = torch.nn.MSELoss()
            if opt.which_perceptual == '5_2':
                self.perceptual_layer = -1
            elif opt.which_perceptual == '4_2':
                self.perceptual_layer = -2

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode, GforD=None):

        generated_out = {}
        if mode == 'generator':
            real_image, input_semantics, ref_image = self.preprocess_input(data.copy(), mode=mode)
            g_loss, generated_out = self.compute_generator_loss(input_semantics, real_image, ref_image)
            
            out = {}
            out['fake_image'] = generated_out['fake_image']
            out['input_semantics'] = input_semantics

            return g_loss, out

        elif mode == 'discriminator':
            real_image, input_semantics, ref_image = self.preprocess_input(data.copy(), mode=mode)
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, ref_image, GforD)
            return d_loss
        elif mode == 'inference':
            input_semantics, ref_image = self.preprocess_input(data.copy(), mode=mode)
            out = {}
            with torch.no_grad():
                out = self.inference(input_semantics, ref_image)
            return out
        else:
            raise ValueError("|mode| is invalid")


    def create_optimizers(self, opt):
        G_params, D_params = list(), list()
        G_params += [{'params': self.net['netG'].parameters(), 'lr': opt.lr * 0.5}]

        if opt.isTrain:
            D_params += list(self.net['netD'].parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), eps=1e-3)
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.net['netG'], 'G', epoch, self.opt)
        util.save_network(self.net['netD'], 'D', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        net = {}
        net['netG'] = networks.define_G(opt)
        net['netD'] = networks.define_D(opt) if opt.isTrain else None

        if not opt.isTrain or opt.continue_train:
            net['netG'] = util.load_network(net['netG'], 'G', opt.epoch, opt)
            if opt.isTrain:
                net['netD'] = util.load_network(net['netD'], 'D', opt.epoch, opt)
            if (not opt.isTrain) and opt.use_ema:
                net['netG'] = util.load_network(net['netG'], 'G_ema', opt.epoch, opt)
        return net

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data, mode):
        if mode == 'inference':
            return data['label'].cuda(), data['ref'].cuda()
        
        return data['image'].cuda(), data['label'].cuda(), data['ref'].cuda()

    def get_ctx_loss(self, source, target):
        contextual_style5_1 = torch.mean(self.contextual_forward_loss(source[-1], target[-1].detach())) * 8
        contextual_style4_1 = torch.mean(self.contextual_forward_loss(source[-2], target[-2].detach())) * 4
        contextual_style3_1 = torch.mean(self.contextual_forward_loss(F.avg_pool2d(source[-3], 2), F.avg_pool2d(target[-3].detach(), 2))) * 2
        if self.opt.use_22ctx:
            contextual_style2_1 = torch.mean(self.contextual_forward_loss(F.avg_pool2d(source[-4], 4), F.avg_pool2d(target[-4].detach(), 4))) * 1
            return contextual_style5_1 + contextual_style4_1 + contextual_style3_1 + contextual_style2_1
        return contextual_style5_1 + contextual_style4_1 + contextual_style3_1

    def compute_generator_loss(self, input_semantics, real_image, ref_image):
        G_losses = {}
        generate_out = self.generate_fake(input_semantics, real_image, ref_image)

        pred_fake, pred_real = self.discriminate(input_semantics, generate_out['fake_image'], real_image, ref_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False) * self.opt.weight_gan

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        fake_features = self.vggnet_fix(generate_out['fake_image'], ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        loss = 0
        for i in range(len(generate_out['real_features'])):
            loss += weights[i] * F.l1_loss(fake_features[i], generate_out['real_features'][i].detach())
        G_losses['fm'] = loss * self.opt.lambda_vgg * self.opt.fm_ratio

        # G_losses['contextual'] = self.get_ctx_loss(fake_features, generate_out['ref_features']) * self.opt.lambda_vgg * self.opt.ctx_w

        G_losses['L1'] = F.l1_loss(generate_out['fake_image'], real_image) * self.opt.lambda_L1
        
        return G_losses, generate_out

    def compute_discriminator_loss(self, input_semantics, real_image, ref_image, GforD):
        D_losses = {}
        with torch.no_grad():
            fake_image = GforD['fake_image'].detach()
            fake_image.requires_grad_()

        pred_fake, pred_real= self.discriminate(input_semantics, fake_image, real_image, ref_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                            for_discriminator=True) * self.opt.weight_gan
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                            for_discriminator=True) * self.opt.weight_gan

        return D_losses

    def generate_fake(self, input_semantics, real_image, ref_image):
        generate_out = {}
        # ref_relu1_1, ref_relu2_1, ref_relu3_1, ref_relu4_1, ref_relu5_1 = self.vggnet_fix(ref_image, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)

        # generate_out['ref_features'] = [ref_relu1_1, ref_relu2_1, ref_relu3_1, ref_relu4_1, ref_relu5_1]
        generate_out['real_features'] = self.vggnet_fix(real_image, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)

        generate_out['fake_image'] = self.net['netG'](ref_image, input_semantics)

        return generate_out

    def inference(self, input_semantics, ref_image):
        generate_out = {}
        generate_out['fake_image'] = self.net['netG'](ref_image, input_semantics)

        return generate_out

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image, ref_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out, _, _ = self.net['netD'](fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def compute_D_seg_loss(self, out, gt):
        fake_seg, real_seg = self.divide_pred([out])
        fake_seg_loss = F.cross_entropy(fake_seg[0][0], gt)
        real_seg_loss = F.cross_entropy(real_seg[0][0], gt)

        down_gt = F.interpolate(gt.unsqueeze(1).float(), scale_factor=0.5, mode='nearest').squeeze().long()
        fake_seg_loss_down = F.cross_entropy(fake_seg[0][1], down_gt)
        real_seg_loss_down = F.cross_entropy(real_seg[0][1], down_gt)

        seg_loss = fake_seg_loss + real_seg_loss + fake_seg_loss_down + real_seg_loss_down
        return seg_loss