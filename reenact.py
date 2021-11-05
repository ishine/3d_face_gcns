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
from lib.mesh_io import read_obj
from gcn_util.utils import init_sampling
import scipy.io as sio
from lipsync3d.utils import get_downsamp_trans


if __name__ == '__main__':
    opt = Options().parse_args()

    downsamp_trans = get_downsamp_trans()

    mat_path = 'renderer/data/data.mat'
    mat_data = sio.loadmat(mat_path)
    exp_base = torch.from_numpy(mat_data['exp_base']).reshape(-1, 3 * 64)
    exp_base = torch.mm(downsamp_trans, exp_base).reshape(-1, 64)
    M = exp_base.t() @ exp_base # 64 x 64
    M += 1e-8 * torch.eye(len(M))
    M = M.inverse() @ exp_base.t() # 64 x (2232 * 3)
    M = M.cuda()
    print(M.shape)
    # create_dir(os.path.join(opt.src_dir, 'reenact_from_mesh'))
    create_dir(os.path.join(opt.src_dir, 'reenact'))
    create_dir(os.path.join(opt.src_dir, 'reenact_bfm_mesh'))
    alpha = torch.load(os.path.join(opt.tgt_dir, 'reference_alpha.pt')).unsqueeze(0).cuda()
    # alpha_list = load_coef(os.path.join(opt.tgt_dir, 'alpha'))
    beta = torch.load(os.path.join(opt.tgt_dir, 'reference_beta.pt')).unsqueeze(0).cuda()
    # beta_list = load_coef(os.path.join(opt.tgt_dir, 'beta'))
    delta_list = load_coef(os.path.join(opt.src_dir, 'reenact_geometry'))
    # delta_list = load_coef(os.path.join(opt.src_dir, 'reenact_delta'))
    # delta_list = load_coef(os.path.join(opt.tgt_dir, 'delta'))
    gamma_list = load_coef(os.path.join(opt.tgt_dir, 'gamma'))
    angle_list = load_coef(os.path.join(opt.tgt_dir, 'rotation'))
    translation_list = load_coef(os.path.join(opt.tgt_dir, 'translation'))
    crop_region_list = load_coef(os.path.join(opt.tgt_dir, 'crop_region'))
    full_image_list = get_file_list(os.path.join(opt.tgt_dir, 'full'))

    opt.data_dir = opt.tgt_dir

    # face_emb_list = load_face_emb(opt.data_dir)
    face_emb = torch.load(os.path.join(opt.tgt_dir, 'reference_face_emb.pt')).unsqueeze(0).cuda()
    face_model = FaceModel(opt=opt, batch_size=1, load_model=True, downsamp_tran=downsamp_trans)

    top, bottom, left, right = get_max_crop_region(crop_region_list)

    for i in tqdm(range(len(delta_list))):
        # alpha = alpha_list[i + opt.offset].unsqueeze(0).cuda()
        # beta = beta_list[i + opt.offset].unsqueeze(0).cuda()
        delta = delta_list[i].reshape(-1, 1).cuda() # (2232 *3) x 1       # 1674 x 1
        delta = M @ delta
        delta = delta.unsqueeze(0)

        idx = i % (opt.offset_end - opt.offset_start ) + opt.offset_start
        gamma = gamma_list[idx].unsqueeze(0).cuda()
        rotation = angle_list[idx].unsqueeze(0).cuda()
        translation = translation_list[idx].unsqueeze(0).cuda()
        # face_emb = face_emb_list[i].unsqueeze(0).cuda()
        crop_region = crop_region_list[idx]
        full_image = cv2.imread(full_image_list[idx])
        H, W, _ = full_image.shape
        empty_image = np.zeros((H, W, 3), np.uint8)
        render, _, _, _, _ = face_model(alpha, delta, beta, rotation, translation, gamma, face_emb, lower=True, is_train=False)
        render = render.squeeze().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        
        rescaled_render = rescale_and_paste(crop_region, empty_image, render)
        rescaled_render = cv2.cvtColor(rescaled_render, cv2.COLOR_RGB2BGR)
        rescaled_render = rescaled_render[top:bottom, left:right]
        rescaled_render = cv2.resize(rescaled_render, (opt.image_width, opt.image_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(opt.src_dir, 'reenact', '%05d.png' % (i+1)), rescaled_render)



        delta = delta_list[i].unsqueeze(0).cuda()
        landmarks = face_model.downsampled_landmarks(alpha, delta, rotation, translation)
        empty_image = 255 * np.ones((256, 256, 3), np.uint8)
        n = landmarks.shape[0]
        for idx in range(n):
            x, y = int(landmarks[idx][0].item()), int(landmarks[idx][1].item())
            empty_image = cv2.circle(empty_image, (x,y), radius=2, color=(0, 0, 0), thickness=-1)

        cv2.imwrite(os.path.join(opt.src_dir, 'reenact_bfm_mesh', '%05d.png' % (i+1)), empty_image)
