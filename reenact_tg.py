import os
import cv2
from tqdm import tqdm
import torch
from torchvision import utils
import numpy as np
from renderer.face_model import FaceModel
from options.options import Options
from audiodvp_utils.util import create_dir, load_coef, get_file_list, get_max_crop_region
from audiodvp_utils.rescale_image import rescale_and_paste
import scipy.io as sio
from lipsync3d.utils import get_downsamp_trans, viterbi_algorithm


if __name__ == '__main__':
    opt = Options().parse_args()

    downsamp_trans = get_downsamp_trans()

    mat_path = 'renderer/data/data.mat'
    mat_data = sio.loadmat(mat_path)
    exp_base = torch.from_numpy(mat_data['exp_base']).reshape(-1, 3 * 64)
    exp_base = torch.mm(downsamp_trans, exp_base).reshape(-1, 64).double()
    reenact_save_path = os.path.join(opt.tgt_dir, 'reenact')
    reenact_crop_lip_path = os.path.join(opt.tgt_dir, 'reenact_crop_lip')
    create_dir(reenact_save_path)
    create_dir(reenact_crop_lip_path)
    alpha_list = load_coef(os.path.join(opt.tgt_dir, 'alpha'))
    beta_list = load_coef(os.path.join(opt.tgt_dir, 'beta'))
    gamma_list = load_coef(os.path.join(opt.tgt_dir, 'gamma'))
    angle_list = load_coef(os.path.join(opt.tgt_dir, 'rotation'))
    translation_list = load_coef(os.path.join(opt.tgt_dir, 'translation'))
    crop_region_list = load_coef(os.path.join(opt.tgt_dir, 'crop_region'))
    full_image_list = get_file_list(os.path.join(opt.tgt_dir, 'full'))
    crop_lip_image_list = get_file_list(os.path.join(opt.tgt_dir, 'crop_lip'))

    opt.data_dir = opt.tgt_dir

    opt.batch_size = 1
    face_model = FaceModel(opt=opt, batch_size=1, load_model=True, downsamp_tran=downsamp_trans)

    top, bottom, left, right = get_max_crop_region(crop_region_list)

    delta_list, crop_lip_indices = viterbi_algorithm(opt.tgt_dir, opt.tg_path)
    
    for i in tqdm(range(len(delta_list))):
        alpha = alpha_list[0].unsqueeze(0).cuda()
        beta = beta_list[0].unsqueeze(0).cuda()
        delta = delta_list[i].unsqueeze(0).cuda()
        crop_lip_image = cv2.imread(crop_lip_image_list[crop_lip_indices[i]])
        
        idx = i % (opt.offset_end - opt.offset_start) + opt.offset_start
        # idx = i
        gamma = gamma_list[idx].unsqueeze(0).cuda()
        rotation = angle_list[idx].unsqueeze(0).cuda()
        translation = translation_list[idx].unsqueeze(0).cuda()
        crop_region = crop_region_list[idx]
        full_image = cv2.imread(full_image_list[idx])
        
        H, W, _ = full_image.shape
        empty_image = np.zeros((H, W, 3), np.uint8)
        render, _, _, _, _ = face_model(alpha, delta, beta, rotation, translation, gamma, lower=True)
        render = render.squeeze().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        
        rescaled_render = rescale_and_paste(crop_region, empty_image, render)
        rescaled_render = cv2.cvtColor(rescaled_render, cv2.COLOR_RGB2BGR)
        rescaled_render = rescaled_render[top:bottom, left:right]
        rescaled_render = cv2.resize(rescaled_render, (opt.image_width, opt.image_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(reenact_save_path, '%05d.png' % (i+1)), rescaled_render)
        cv2.imwrite(os.path.join(reenact_crop_lip_path, '%05d.png' % (i+1)), crop_lip_image)
