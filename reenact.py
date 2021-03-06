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
from audiodvp_utils.mouth_retrieval import mouth_retrieval


if __name__ == '__main__':
    opt = Options().parse_args()

    create_dir(os.path.join(opt.src_dir, 'reenact'))
    create_dir(os.path.join(opt.src_dir, 'reenact_crop_lip'))
    create_dir(os.path.join(opt.src_dir, 'reenact_landmarks'))

    alpha_list = load_coef(os.path.join(opt.tgt_dir, 'alpha'))
    beta_list = load_coef(os.path.join(opt.tgt_dir, 'beta'))
    tgt_delta_list = load_coef(os.path.join(opt.tgt_dir, 'delta'))
    src_delta_list = load_coef(os.path.join(opt.src_dir, 'delta'))
    src_alpha_list = load_coef(os.path.join(opt.tgt_dir, 'alpha'))
    gamma_list = load_coef(os.path.join(opt.tgt_dir, 'gamma'))
    angle_list = load_coef(os.path.join(opt.tgt_dir, 'rotation'))
    translation_list = load_coef(os.path.join(opt.tgt_dir, 'translation'))
    crop_region_list = load_coef(os.path.join(opt.tgt_dir, 'crop_region'))
    full_image_list = get_file_list(os.path.join(opt.tgt_dir, 'full'))
    masks = get_file_list(os.path.join(opt.tgt_dir, 'mask'))
    tgt_crop_lip_image_list = get_file_list(os.path.join(opt.tgt_dir, 'crop_lip'))
    src_crop_lip_image_list = get_file_list(os.path.join(opt.src_dir, 'crop_lip'))
    opt.data_dir = opt.tgt_dir

    tgt_crop_lip_h = torch.load(os.path.join(opt.tgt_dir, 'crop_lip_h.pt'))
    src_crop_lip_h = torch.load(os.path.join(opt.src_dir, 'crop_lip_h.pt'))
    retrieval = mouth_retrieval(opt, tgt_delta_list, tgt_crop_lip_h, tgt_crop_lip_image_list)
    
    opt.batch_size = 1
    transfer = transformation(opt, src_alpha_list[0], alpha_list[0])
    face_model = FaceModel(opt=opt, batch_size=1, load_model=True)

    top, bottom, left, right = get_max_crop_region(crop_region_list)

    for i in tqdm(range(len(src_delta_list))):
        alpha = alpha_list[0].unsqueeze(0).cuda()
        beta = beta_list[0].unsqueeze(0).cuda()
        delta = src_delta_list[i]
        new_delta = transfer.deformation_transfer(delta)
        crop_lip_index = retrieval.retrieve(new_delta, src_crop_lip_image_list[i], src_crop_lip_h[i])
        crop_lip_image = cv2.imread(tgt_crop_lip_image_list[crop_lip_index])
        
        new_delta = new_delta.unsqueeze(0).cuda()
        
        # idx = i % (opt.offset_end - opt.offset_start ) + opt.offset_start
        idx = i % len(full_image_list)
        gamma = gamma_list[idx].unsqueeze(0).cuda()
        rotation = angle_list[idx].unsqueeze(0).cuda()
        translation = translation_list[idx].unsqueeze(0).cuda()
        crop_region = crop_region_list[idx]
        full = cv2.imread(full_image_list[idx])
        mask = cv2.imread(masks[idx]) / 255.0
        H, W, _ = full.shape
        empty_image = np.zeros((H, W, 3), np.uint8)
        render, _, tmp_landmarks, _, _ = face_model(alpha, new_delta, beta, rotation, translation, gamma)
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
        
        # # Convert landmarks to numpy array
        # tmp_landmarks = tmp_landmarks.cpu()
        # shape_np = np.zeros((68, 2), dtype="int")
        # for idx in range(0, 68):
        #     landmark = tmp_landmarks[0][idx]
        #     shape_np[idx] = (landmark[0], landmark[1])

        # # Draw landmarks
        # for idx, (x, y) in enumerate(shape_np):
        #     render_landmark = cv2.circle(render, (x, y), 3, (255, 0, 0), -1)
                    
        # # Landmark frames
        # rescaled_render_landmark = rescale_and_paste(crop_region, empty_image, render_landmark)
        # rescaled_render_landmark = cv2.cvtColor(rescaled_render_landmark, cv2.COLOR_RGB2BGR)
        # rescaled_render_landmark = rescaled_render_landmark[top:bottom, left:right]
        # rescaled_render_landmark = cv2.resize(rescaled_render_landmark, (opt.image_width, opt.image_height), interpolation=cv2.INTER_AREA)
        
        # cv2.imwrite(os.path.join(opt.src_dir, 'reenact_landmarks', '%05d.png' % (i+1)), rescaled_render_landmark)
