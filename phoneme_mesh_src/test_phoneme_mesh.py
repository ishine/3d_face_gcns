import sys
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import torch
from torch.utils.data import DataLoader
from options import Options
from dataset import Lipsync3DDataset
from loss import L2Loss
import time
from utils import mesh_tensor_to_landmarkdict, draw_mesh_images
import os
from tqdm import tqdm
import cv2

from phoneme_mesh import PhonemeMeshModel

if __name__ == '__main__':
    opt = Options().parse_args()
    calculate_test_loss = (opt.src_dir == opt.tgt_dir)
    dataset = Lipsync3DDataset(opt)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,  # default not shuffle
        num_workers=opt.num_workers,
        drop_last=True  # the batch size cannot change during the training so the last uncomplete batch need to be dropped
    )
    criterionGeo = L2Loss()
    os.makedirs(os.path.join(opt.src_dir, 'reenact_mesh'), exist_ok=True)
    os.makedirs(os.path.join(opt.src_dir, 'reenact_mesh_image'), exist_ok=True)
    os.makedirs(os.path.join(opt.src_dir, 'reenact_texture'), exist_ok=True)
    os.makedirs(os.path.join(opt.src_dir, 'predicted_normalised_mesh'), exist_ok=True)

    model = PhonemeMeshModel()
    model.load('phoneme_mesh_model_middle.pt')
    
    avg_loss = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader)):
            reference_mesh = data['reference_mesh']
            normalized_mesh = data['normalized_mesh']
            filename = data['filename'][0]
            R = data['R'][0]
            RT = R.transpose(0, 1)
            t = data['t'][0]
            c = data['c'][0]

            if opt.use_src_seq:
                src_seq = data['src_seq']
            else:
                print("please use src seq!")
                break
            B, _ = src_seq.shape
            geo_diff_list = []
            for bi in range(B):
                pho = src_seq[bi, 12].item()
                geo_diff_list.append(model.recall(pho))
            geometry_diff = torch.stack(geo_diff_list, dim=0)

            geometry = reference_mesh + geometry_diff

            if calculate_test_loss and (i > int(len(test_dataloader) * opt.train_rate)):
                geoLoss = criterionGeo(geometry, normalized_mesh)
                avg_loss += geoLoss / int(len(test_dataloader) * (1 - opt.train_rate))

            geometry = geometry[0].transpose(0, 1)
            normlaised_geometry = geometry.clone().detach()
            normalised_landmark_dict = mesh_tensor_to_landmarkdict(normlaised_geometry)
            
            geometry = (torch.matmul(RT, (geometry - t)) / c).transpose(0, 1)
            landmark_dict = mesh_tensor_to_landmarkdict(geometry)

            torch.save(normalised_landmark_dict, os.path.join(opt.src_dir,'predicted_normalised_mesh',filename))
            torch.save(landmark_dict, os.path.join(opt.src_dir, 'reenact_mesh', filename))
    
    if calculate_test_loss:
        print('Average Test loss : ', avg_loss)

    print('Start drawing reenact mesh')
    image = cv2.imread(os.path.join(opt.tgt_dir, 'reference_frame.png'))
    image_rows, image_cols, _ = image.shape
    draw_mesh_images(os.path.join(opt.src_dir, 'reenact_mesh'), os.path.join(opt.src_dir, 'reenact_mesh_image'), image_rows, image_cols)
            

