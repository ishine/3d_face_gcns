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

if __name__ == '__main__':
    opt = Options().parse_args()

    create_dir(os.path.join(opt.src_dir, 'reenact'))

    alpha_list = load_coef(os.path.join(opt.tgt_dir, 'alpha'))
    beta_list = load_coef(os.path.join(opt.tgt_dir, 'beta'))
#    delta_list = load_coef(os.path.join(opt.src_dir, 'reenact_delta'))
    delta_list = load_coef(os.path.join(opt.src_dir, 'delta'))
    gamma_list = load_coef(os.path.join(opt.tgt_dir, 'gamma'))
    angle_list = load_coef(os.path.join(opt.tgt_dir, 'rotation'))
    translation_list = load_coef(os.path.join(opt.tgt_dir, 'translation'))
    crop_region_list = load_coef(os.path.join(opt.tgt_dir, 'crop_region'))
    full_image_list = get_file_list(os.path.join(opt.tgt_dir, 'full'))

    opt.data_dir = opt.tgt_dir

    face_emb_list = load_face_emb(opt.data_dir)
    face_model = FaceModel(opt=opt, batch_size=1, load_model=True)

    top, bottom, left, right = get_max_crop_region(crop_region_list)

    for i in tqdm(range(min(len(delta_list), len(alpha_list)))):
        alpha = alpha_list[i + opt.offset].unsqueeze(0).cuda()
        beta = beta_list[i + opt.offset].unsqueeze(0).cuda()
        delta = delta_list[i].unsqueeze(0).cuda()
        gamma = gamma_list[i + opt.offset].unsqueeze(0).cuda()
        rotation = angle_list[i + opt.offset].unsqueeze(0).cuda()
        translation = translation_list[i + opt.offset].unsqueeze(0).cuda()
        face_emb = face_emb_list[i].unsqueeze(0).cuda()
        crop_region = crop_region_list[i]
        full_image = cv2.imread(full_image_list[i])
        H, W, _ = full_image.shape
        empty_image = np.zeros((H, W, 3), np.uint8)
        render, _, _, _, _ = face_model(alpha, delta, beta, rotation, translation, gamma, face_emb, lower=True)
        render = render.squeeze().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        
        rescaled_render = rescale_and_paste(crop_region, empty_image, render)
        rescaled_render = cv2.cvtColor(rescaled_render, cv2.COLOR_RGB2BGR)
        rescaled_render = rescaled_render[top:bottom, left:right]
        rescaled_render = cv2.resize(rescaled_render, (opt.image_width, opt.image_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(opt.src_dir, 'reenact', os.path.basename(full_image_list[i])), rescaled_render)

        if i >= opt.test_num:
            break
