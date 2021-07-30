import os
from collections import OrderedDict
import torch
from torchvision import utils
import torch.nn.functional as F
import numpy as np
from . import networks
from audiodvp_utils.util import create_dir
from facenet_pytorch import InceptionResnetV1
from .discriminator import Discriminator

class ResnetModel:
    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self.isTrain = opt.isTrain

        self.epoch_tex = 5
        self.epoch_warm_up = 10
        self.epoch_train = opt.num_epoch
        self.pass_epoch_tex = False
        self.pass_epoch_warm = False


        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['Photometric', 'Landmark', 'Alpha', 'Beta', 'Delta', 'Identity', 'Vertex', 'Var', 'Adv_G', 'Adv_D']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['gt', 'render', 'masked_gt', 'overlay']

        # define networks (both generator and discriminator)
        self.net = networks.ResnetFaceModelOptimizer(opt)
        # self.discriminator = Discriminator().to(self.device)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.upsample = torch.nn.Upsample((224, 224))

        if self.isTrain:
            # define loss functions
            self.criterionPhotometric = networks.PhotometricLoss() if self.opt.lambda_photo > 0.0 else None
            self.criterionLandmark = networks.LandmarkLoss(opt) if self.opt.lambda_land > 0.0 else None
            self.regularizationAlpha = networks.CoefficientRegularization()
            self.regularizationBeta = networks.CoefficientRegularization()
            self.regularizationDelta = networks.CoefficientRegularization()

            # face-gcn loss function
            self.criterionVertex = networks.VertexWiseLoss()
            self.criterionIdentity = networks.IdentityPreservingLoss()
            # self.criterionAdv = networks.AdversarialLoss()
            self.criterionVar = networks.VarianceLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam([{'params' : self.net.fc.parameters()}, 
                                                {'params' : self.net.pretrained_model.parameters()}, 
                                                {'params' : self.net.face_model.refiner.parameters()}],
                                               lr=opt.lr, betas=(0.5, 0.999)
            )

            # self.optimizer_D = torch.optim.Adam([{'params' : self.discriminator.parameters()}],
            #                                    lr=opt.lr, betas=(0.5, 0.999)
            # )

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.input = input['input'].to(self.device)
        self.gt = input['gt'].to(self.device)
        self.landmark_gt = input['landmark_gt'].to(self.device)
        self.face_emb = input['face_emb'].reshape(self.opt.batch_size, -1).to(self.device)
        self.image_name = input['image_name']

    def forward(self, epoch=None):
        if (epoch != None) and (epoch < self.epoch_tex):
            self.alpha, self.delta, self.beta, self.gamma, self.rotation, self.translation, self.render, self.mask, self.landmark, self.tex, self.refined_tex = self.net(self.input, None)
        else:
            self.alpha, self.delta, self.beta, self.gamma, self.rotation, self.translation, self.render, self.mask, self.landmark, self.tex, self.refined_tex = self.net(self.input, self.face_emb)
        
        self.masked_gt = self.gt * self.mask
        self.overlay = self.render + (1 - self.mask) * self.gt
        self.upsampled_overlay = self.upsample(self.overlay)
        self.render_emb = self.facenet(self.upsampled_overlay)


    def backward_G(self, epoch):
        self.loss_Photometric = self.criterionPhotometric(self.render, self.masked_gt) if self.opt.lambda_photo > 0.0 else 0.0
        self.loss_Landmark = self.criterionLandmark(self.landmark, self.landmark_gt) if self.opt.lambda_land > 0.0 else 0.0

        self.loss_Alpha = self.regularizationAlpha(self.alpha)
        self.loss_Beta = self.regularizationBeta(self.beta)
        self.loss_Delta = self.regularizationDelta(self.delta)

        self.loss_Vertex = self.criterionVertex(self.tex, self.refined_tex)
        self.loss_Identity = 0 #self.criterionIdentity(self.face_emb, self.render_emb)
        self.loss_Var = 0 #self.criterionVar(self.refined_tex, self.net.face_model.skin_index)

        # self.fake_score = self.discriminator(self.upsampled_overlay)
        # self.loss_Adv_G = self.criterionAdv(self.fake_score, True)
        self.loss_Adv_G = 0
        self.loss_Adv_D = 0

        ren_lambda = np.clip(0.2 * (epoch - self.epoch_tex), 0, 1).astype(np.float32) if epoch >= self.epoch_tex else 1
        ref_lambda = np.clip(1 - ren_lambda, 0, 1).astype(np.float32)

        # if epoch >= self.epoch_tex:
        #     self.opt.lambda_id = 0.2
        #     self.opt.lambda_var = 5

        # combine loss and calculate gradients
        self.loss = self.opt.lambda_photo * self.loss_Photometric \
            + self.opt.lambda_reg * \
            (self.opt.lambda_alpha * self.loss_Alpha + self.opt.lambda_beta * self.loss_Beta + self.opt.lambda_delta * self.loss_Delta) \
            + self.opt.lambda_land * self.loss_Landmark \
            + self.opt.lambda_var * self.loss_Var \
            + self.opt.lambda_id * self.loss_Identity
        
        self.loss = ren_lambda * (self.loss  + self.opt.lambda_adv * self.loss_Adv_G) \
                    + ref_lambda * self.loss_Vertex

        self.loss.backward()

    
    def backward_D(self, epoch):
        self.real_score = self.discriminator(self.input)
        self.fake_score = self.discriminator(self.upsampled_overlay)

        self.loss_Adv_D = (self.criterionAdv(self.real_score, True) + self.criterionAdv(self.fake_score, False)) * 0.5
        gradient_penalty, _ = networks.cal_gradient_penalty(self.discriminator, self.input, self.upsampled_overlay, 'cuda', type='mixed', constant=1.0, lambda_gp=10.0)

        self.loss = (self.loss_Adv_D + gradient_penalty) * self.opt.lambda_adv
        self.loss.backward(retain_graph=True)

    def optimize_parameters(self, epoch):
        if epoch == self.epoch_tex and (not self.pass_epoch_tex):
            self.set_requires_grad(self.net.fc, False)
            self.set_requires_grad(self.net.pretrained_model, False)
            self.net.fc.eval()
            self.net.pretrained_model.eval()
            self.pass_epoch_tex = True
        elif epoch == self.epoch_warm_up and (not self.pass_epoch_warm):
            self.set_requires_grad(self.net.fc, True)
            self.set_requires_grad(self.net.pretrained_model, True)
            self.net.fc.train()
            self.net.pretrained_model.train()
            self.pass_epoch_warm = True
            
        self.forward(epoch)

        # self.set_requires_grad(self.discriminator, True)  # enable backprop for D
        # self.optimizer_D.zero_grad()     # set D's gradients to zero
        # self.backward_D(epoch)                # calculate gradients for D
        # self.optimizer_D.step()          # update D's weights
        # # update G
        # self.set_requires_grad(self.discriminator, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G(epoch)                   # calculate graidents for G
        self.optimizer_G.step()

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

    def save_result(self):
        """Save 3DMM coef and image"""
        create_dir(os.path.join(self.opt.data_dir, 'render'))
        create_dir(os.path.join(self.opt.data_dir, 'overlay'))
        create_dir(os.path.join(self.opt.data_dir, 'alpha'))
        create_dir(os.path.join(self.opt.data_dir, 'beta'))
        create_dir(os.path.join(self.opt.data_dir, 'delta'))
        create_dir(os.path.join(self.opt.data_dir, 'gamma'))
        create_dir(os.path.join(self.opt.data_dir, 'rotation'))
        create_dir(os.path.join(self.opt.data_dir, 'translation'))
        create_dir(os.path.join(self.opt.data_dir, 'refinement'))

        for i in range(self.opt.batch_size):
            utils.save_image(self.render[i], os.path.join(self.opt.data_dir, 'render', self.image_name[i]))
            utils.save_image(self.overlay[i], os.path.join(self.opt.data_dir, 'overlay', self.image_name[i]))

            torch.save(self.alpha[i].detach().cpu(), os.path.join(self.opt.data_dir, 'alpha', self.image_name[i][:-4]+'.pt'))
            torch.save(self.beta[i].detach().cpu(), os.path.join(self.opt.data_dir, 'beta', self.image_name[i][:-4]+'.pt'))
            torch.save(self.delta[i].detach().cpu(), os.path.join(self.opt.data_dir, 'delta', self.image_name[i][:-4]+'.pt'))
            torch.save(self.gamma[i].detach().cpu(), os.path.join(self.opt.data_dir, 'gamma', self.image_name[i][:-4]+'.pt'))
            torch.save(self.rotation[i].detach().cpu(), os.path.join(self.opt.data_dir, 'rotation', self.image_name[i][:-4]+'.pt'))
            torch.save(self.translation[i].detach().cpu(), os.path.join(self.opt.data_dir, 'translation', self.image_name[i][:-4]+'.pt'))

        torch.save(self.net.face_model.refiner.state_dict(), os.path.join(self.opt.data_dir, 'refinement', 'model.pt'))

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
