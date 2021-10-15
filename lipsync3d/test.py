import sys
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import torch
from torch.utils.data import DataLoader
from options import Options
from dataset import Lipsync3DDataset
from model import Lipsync3DModel
from loss import L2Loss, L1Loss
import time
from utils import mesh_tensor_to_landmarkdict, draw_mesh_images
import os
from tqdm import tqdm
import cv2

if __name__ == '__main__':
    opt = Options().parse_args()
    device = opt.device
    calculate_test_loss = (opt.src_dir == opt.tgt_dir)
    dataset = Lipsync3DDataset(opt)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,  # default not shuffle
        num_workers=opt.num_workers,
        drop_last=True  # the batch size cannot change during the training so the last uncomplete batch need to be dropped
    )

    model = Lipsync3DModel().to(device)
    criterionGeo = L2Loss()
    criterionTex = L1Loss()
    ckpt = torch.load(os.path.join(opt.tgt_dir, 'Lipsync3dnet.pth'), map_location=device)
    model.load_state_dict(ckpt)
    os.makedirs(os.path.join(opt.src_dir, 'reenact_mesh'), exist_ok=True)
    os.makedirs(os.path.join(opt.src_dir, 'reenact_mesh_image'), exist_ok=True)
    
    avg_loss = 0
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(tqdm(test_dataloader)):
            audio_feature = data['audio_feature'].to(device)
            reference_mesh = data['reference_mesh'].to(device)
            normalized_mesh = data['normalized_mesh'].to(device)
            filename = data['filename'][0]
            R = data['R'][0].to(device)
            RT = R.transpose(0, 1)
            t = data['t'][0].to(device)
            c = data['c'][0].to(device)

            # geometry_diff, texture_dfif = model(audio_feature)
            geometry_diff = model(audio_feature)
            geometry_diff = geometry_diff.reshape(-1, 478, 3)
            geometry = reference_mesh + geometry_diff

            if calculate_test_loss and (i > int(len(test_dataloader) * opt.train_rate)):
                geoLoss = criterionGeo(geometry, normalized_mesh)
                avg_loss += geoLoss.detach() / int(len(test_dataloader) * (1 - opt.train_rate))

            geometry = geometry[0].transpose(0, 1)
            geometry = (torch.matmul(RT, (geometry - t)) / c).transpose(0, 1).cpu().detach()
            landmark_dict = mesh_tensor_to_landmarkdict(geometry)

            torch.save(landmark_dict, os.path.join(opt.src_dir, 'reenact_mesh', filename))
    
    if calculate_test_loss:
        print('Average Test loss : ', avg_loss)

    print('Start drawing reenact mesh')
    image = cv2.imread(os.path.join(opt.tgt_dir, 'reference_frame.png'))
    image_rows, image_cols, _ = image.shape
    draw_mesh_images(os.path.join(opt.src_dir, 'reenact_mesh'), os.path.join(opt.src_dir, 'reenact_mesh_image'), image_rows, image_cols)
            

