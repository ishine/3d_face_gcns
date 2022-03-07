from audiodvp_utils.util import create_dir, get_file_list
import os
import cv2
from matplotlib import pyplot as plt
import face_alignment
from tqdm import tqdm
from skimage import io
from models import networks
import torch
from renderer.face_model import FaceModel
from torchvision import transforms, utils
import time
import scipy.io as sio
import numpy as np


def landmark_detection(image_name):
    image = io.imread(image_name)
    fa_3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cuda:1')
    preds = fa_3d.get_landmarks(image)
    
    return preds[0][:, :2]
   
        
class IRLSSolver():
    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self.data_dir = opt.data_dir
        self.fa_3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=self.device)
        self.opt = opt
        self.batch_size = 1
        self.face_model = FaceModel(opt=opt, batch_size=1, image_width=256, image_height=256)
        
        self.criterionPhotometric = networks.PhotometricLoss(mode='None')
        self.criterionLandmark = networks.LandmarkLoss(self.device, mode='None')
        self.regularizationAlpha = networks.CoefficientRegularization(mode='None')
        self.regularizationBeta = networks.CoefficientRegularization(mode='None')
        self.regularizationDelta = networks.CoefficientRegularization(mode='None')
        
        self.lambda_photo =  self.opt.lambda_photo ** (1/2)   #1.9 ** (1/2) 
        self.lambda_land = self.opt.lambda_land ** (1/2) # 1.6e-3 ** (1/2)
        self.lambda_reg = self.opt.lambda_reg ** (1/2) #3e-4 ** (1/2)
        self.lambda_alpha = self.opt.lambda_alpha * self.lambda_reg       # 1.0
        self.lambda_beta = self.opt.lambda_beta * self.lambda_reg    #1.7e-3 * self.lambda_reg
        self.lambda_delta = self.opt.lambda_delta * self.lambda_reg #0.8 * self.lambda_reg       
        
        
        self.transform = transforms.ToTensor()
        
        self.GN_iter = 30
        self.PCG_iter = 4
        
        self.diag = torch.ones(257, device = self.device) * 0.01
        self.prev_g = None
        self.params_diff = None
        self.prev_residual = None
    
    
    # compute torch.mm(j^T, v)
    def vjp(self, output, input, v):
        # start = time.time()
        result = torch.autograd.grad(output, input, grad_outputs=v, retain_graph=True)[0]
        # print('vjp time: ', time.time()-start)
        return result
    
    
    # compute torch.mm(j, v)
    def jvp(self, output, input, v):
        # start = time.time()
        grad_output = torch.zeros_like(output, requires_grad=True)
        grad_input = torch.autograd.grad(output, input, grad_outputs=grad_output, create_graph=True)[0]
        grad_res = torch.autograd.grad(grad_input, grad_output, grad_outputs=v)[0]
        # print('jvp time: ', time.time()-start)
        return grad_res
    
    
    def jtjvp(self, output, input, v):
        jvp_res = self.jvp(output, input, v)
        grad_res = self.vjp(output, input, jvp_res)
        
        return grad_res
    
    
    # def calculate_jacobian(self, input, output):
    #     jacobian = torch.zeros((output.shape[1], input.shape[1]), device=self.device)
        
    #     for i in tqdm(range(int(output.shape[1] / self.batch_size))):
    #         weight = torch.zeros((self.batch_size, output.shape[1]), device=self.device)
    #         batch_idx = torch.arange(0, self.batch_size)
    #         idx = torch.arange(i * self.batch_size, (i+1) * self.batch_size)
    #         weight[batch_idx, idx] = 1.0
    #         output.backward(weight, retain_graph=True)
    #         jacobian[i*self.batch_size:(i+1)*self.batch_size, :] = input.grad.data.clone().squeeze(-1)

    #         input.grad.zero_()
        
    #     if (output.shape[1] % self.batch_size) != 0:
    #         num_remain = output.shape[1] % self.batch_size
    #         weight = torch.zeros((self.batch_size, output.shape[1]), device=self.device)
    #         batch_idx = torch.arange(0, num_remain)
    #         idx = torch.arange(output.shape[1] - num_remain, output.shape[1])
    #         weight[batch_idx, idx] = 1.0
    #         output.backward(weight, retain_graph=True)
    #         jacobian[output.shape[1] - num_remain:output.shape[1], :] = input.grad.data.clone().squeeze(-1)[:num_remain, :]
    #         input.grad.zero_()
            
    #     self.jacobian = jacobian
        
    #     return self.jacobian
    
    
    # def get_diagonal(self, image, landmark_gt, params):
        
    #     params = params.clone().detach()
    #     params = params.expand(self.batch_size, -1, -1)
    #     params.requires_grad = True
        
    #     _, _, loss = self.get_output(image, landmark_gt, params, is_diag=True)

    #     diag_elem = torch.zeros(257, device=self.device)
        
    #     remain = 257
    #     for i in tqdm(range(int(257 / self.batch_size) + 1)):
    #         weight = torch.zeros((self.batch_size, 257, 1), device=self.device)
    #         batch_idx = torch.arange(0, min(self.batch_size, remain))
    #         idx = torch.arange(i * self.batch_size, i * self.batch_size + min(self.batch_size, remain))
    #         weight[batch_idx, idx] = 1.0
    #         batch_diag = self.jvp(loss, params, weight)[:remain]
    #         batch_diag = (batch_diag ** 2).sum(dim=1).reshape(-1)
    #         diag_elem[idx] = batch_diag
    #         remain -= self.batch_size
        
    #     return torch.diag(diag_elem)
    
    
    def get_approx_diagonal(self, b, loss):
        if self.params_diff == None:
            self.prev_g = -b
            self.prev_residual = loss
            return torch.diag(self.diag)
        
        curr_g = -b
        u = curr_g - self.prev_g
        diag_diff = (((2) * (self.params_diff * u + self.prev_g * self.params_diff) \
                    + ((loss ** 2).sum() - 2 * (self.params_diff * u).sum() - (self.params_diff * self.prev_g).sum()) * (self.params_diff * self.params_diff)) / (self.params_diff ** 2).sum()).squeeze()
        
        temp = self.diag + diag_diff
        temp[temp < 0] = self.diag[temp < 0]
        self.diag = temp
        self.prev_g = -b
        self.prev_residual = loss
        
        return torch.diag(self.diag)

        
    
    def PCG_step(self, image, landmark_gt, params):
        render, mask, loss = self.get_output(image, landmark_gt, params, verbose=True)
        b = self.vjp(loss, params, -loss.clone().detach())[0].detach()
        
        M = self.get_approx_diagonal(b, loss.clone().detach())
        
        x_prev = torch.zeros((257, 1), device=self.device)
        r_prev = b - self.jtjvp(loss, params, x_prev.unsqueeze(0))[0]
        M_inv = torch.inverse(M)
        z_prev = M_inv.mm(r_prev)
        p_curr = z_prev
        w = self.jtjvp(loss, params, p_curr.unsqueeze(0))[0]
        alpha_curr = (torch.mm(r_prev.T, z_prev) / torch.mm(p_curr.T, w)).item()
        x_curr = x_prev + alpha_curr * p_curr
        r_curr = r_prev - alpha_curr * w

        for _ in range(self.PCG_iter):
            z_curr = M_inv.mm(r_curr)
            beta_curr = (torch.mm(r_curr.T, z_curr) / torch.mm(r_prev.T, z_prev)).item()
            p_next = z_curr + beta_curr * p_curr
            w = self.jtjvp(loss, params, p_next.unsqueeze(0))[0]
            alpha_next = (torch.mm(r_curr.T, z_curr) / torch.mm(p_next.T, w)).item()
            x_next = x_curr + alpha_next * p_next
            r_next = r_curr - alpha_next * w
            
            x_curr = x_next
            z_prev = z_curr
            r_prev = r_curr
            r_curr = r_next
        
        return render, mask, x_next
    
    
    def update(self, image, landmark_gt, params):
        render, mask, self.params_diff = self.PCG_step(image, landmark_gt, params)
        new_params = (params + self.params_diff).detach()
        new_params.requires_grad = True
        return render, mask, new_params
    
    
    def get_parameters(self, image_name):
        image = io.imread(image_name)
        landmark_gt = torch.tensor(self.fa_3d.get_landmarks(image)[0][:, :2]).to(self.device)
        params = torch.zeros((1, 257, 1), requires_grad=True, device=self.device)
        image = self.transform(image).to(self.device)
        
        save_dir = os.path.join(self.data_dir, "solver_test")
        create_dir(save_dir)
        
        for i in range(self.GN_iter):
            print('step {}'.format(i))
            render, mask, params = self.update(image, landmark_gt, params)
            overlay = image * (1 - mask[0]) + render[0] * mask[0]
            utils.save_image(overlay, os.path.join(save_dir, "test_img_{}.png".format(i)))
        
        return render, mask
    
    
    def get_output(self, image, landmark_gt, params, verbose=False):
        batch_size = 1
        alpha = params[:, :80]
        beta = params[:, 80:160]
        delta = params[:, 160:224]
        rotation = params[:, 224:227]
        translation = params[:, 227:230]
        gamma = params[:, 230:]
        
        render, mask, landmark, _, _ = self.face_model(alpha, delta, beta, rotation, translation, gamma, use_refine=False)

        masked_image = image * mask[0]
        x, y = (mask[0][0] == 1).nonzero(as_tuple=True)
        self.loss_Photometric = self.criterionPhotometric(render, masked_image)[:, :, x, y].reshape(batch_size, -1) / (mask[0].sum()**(1/2))
        self.loss_Landmark = self.criterionLandmark(landmark, landmark_gt).reshape(batch_size, -1)
        self.loss_Alpha = self.regularizationAlpha(alpha).squeeze(-1)
        self.loss_Beta = self.regularizationBeta(beta).squeeze(-1)
        self.loss_Delta = self.regularizationDelta(delta).squeeze(-1)
        
        loss = torch.cat([self.lambda_photo * self.loss_Photometric,
                        self.lambda_land * self.loss_Landmark, 
                        self.lambda_alpha * self.loss_Alpha,
                        self.lambda_beta * self.loss_Beta,
                        self.lambda_delta * self.loss_Delta], dim=1)

        if verbose:
            print('Photo: {}, land: {}, alpha: {}, Beta: {}, delta: {}'.format(torch.sum(self.loss_Photometric[0] ** 2),
                                                                            torch.sum(self.loss_Landmark[0] ** 2), 
                                                                            torch.sum(self.loss_Alpha[0] ** 2),
                                                                            torch.sum(self.loss_Beta[0] ** 2), 
                                                                            torch.sum(self.loss_Delta[0] ** 2)))
        
        return render, mask, loss
        
    
    
    