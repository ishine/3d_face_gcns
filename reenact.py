import os
import cv2
from tqdm import tqdm
import torch
from torchvision import utils
import numpy as np
from renderer.face_model import FaceModel
from options.options import Options
from audiodvp_utils.util import create_dir, load_coef, load_face_emb, get_file_list, get_max_crop_region
from audiodvp_utils.rescale_image import rescale_and_paste
import scipy.io as sio
from audiodvp_utils.deformation_transfer import transformation


if __name__ == '__main__':
    opt = Options().parse_args()

    create_dir(os.path.join(opt.src_dir, 'reenact'))
    create_dir(os.path.join(opt.src_dir, 'reenact_crop_lip'))
    alpha_list = load_coef(os.path.join(opt.tgt_dir, 'alpha'))
    beta_list = load_coef(os.path.join(opt.tgt_dir, 'beta'))
    src_delta_list = load_coef(os.path.join(opt.src_dir, 'delta'))
    src_alpha_list = load_coef(os.path.join(opt.src_dir, 'alpha'))
    gamma_list = load_coef(os.path.join(opt.tgt_dir, 'gamma'))
    angle_list = load_coef(os.path.join(opt.tgt_dir, 'rotation'))
    translation_list = load_coef(os.path.join(opt.tgt_dir, 'translation'))
    crop_region_list = load_coef(os.path.join(opt.tgt_dir, 'crop_region'))
    full_image_list = get_file_list(os.path.join(opt.tgt_dir, 'full'))
    masks = get_file_list(os.path.join(opt.tgt_dir, 'mask'))
    crop_lip_image_list = get_file_list(os.path.join(opt.tgt_dir, 'crop_lip'))

    opt.data_dir = opt.tgt_dir

    opt.batch_size = 1
    transfer = transformation(opt, src_alpha_list[0], alpha_list[0])
    face_model = FaceModel(opt=opt, batch_size=1, load_model=True)

    top, bottom, left, right = get_max_crop_region(crop_region_list)

    for i in tqdm(range(len(src_delta_list))):
        alpha = alpha_list[0].unsqueeze(0).cuda()
        beta = beta_list[0].unsqueeze(0).cuda()
        delta = src_delta_list[i]
        new_delta = transfer.deformation_transfer(delta).unsqueeze(0).cuda() 
        crop_lip_image = cv2.imread(crop_lip_image_list[i])
        
        # idx = i % (opt.offset_end - opt.offset_start ) + opt.offset_start
        idx = i
        gamma = gamma_list[idx].unsqueeze(0).cuda()
        rotation = angle_list[idx].unsqueeze(0).cuda()
        translation = translation_list[idx].unsqueeze(0).cuda()
        crop_region = crop_region_list[idx]
        full = cv2.imread(full_image_list[idx])
        mask = cv2.imread(masks[idx]) / 255.0
        H, W, _ = full.shape
        empty_image = np.zeros((H, W, 3), np.uint8)
        render, _, _, _, _ = face_model(alpha, new_delta, beta, rotation, translation, gamma)
        render = render.squeeze().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        
        rescaled_render = rescale_and_paste(crop_region, empty_image, render)
        rescaled_render = cv2.cvtColor(rescaled_render, cv2.COLOR_RGB2BGR)
        rescaled_render = rescaled_render[top:bottom, left:right]
        rescaled_render = cv2.resize(rescaled_render, (opt.image_width, opt.image_height), interpolation=cv2.INTER_AREA)
        
        rescaled_crop = full[top:bottom, left:right]
        rescaled_crop = cv2.resize(rescaled_crop, (opt.image_width, opt.image_height), interpolation=cv2.INTER_AREA)
        
        masked_crop = rescaled_crop * (1 - mask)
        masked_render = rescaled_render * mask
        masked_image = masked_crop + masked_render
        
        cv2.imwrite(os.path.join(opt.src_dir, 'reenact', '%05d.png' % (i+1)), masked_image)
        cv2.imwrite(os.path.join(opt.src_dir, 'reenact_crop_lip', '%05d.png' % (i+1)), crop_lip_image)
