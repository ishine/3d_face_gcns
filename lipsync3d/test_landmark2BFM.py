import sys
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import torch
from torch.utils.data import DataLoader
from options import Options
from dataset import Landmark2BFMDataset
from model import landmark_to_BFM
from loss import calculate_geoLoss
import time
import os
from tqdm import tqdm

if __name__ == '__main__':
    opt = Options().parse_args()
    device = opt.device
    opt.mesh_dir = os.path.join(opt.src_dir, 'reenact_norm_mesh')
    dataset = Landmark2BFMDataset(opt)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,  # default not shuffle
        num_workers=opt.num_workers,
        drop_last=True  # the batch size cannot change during the training so the last uncomplete batch need to be dropped
    )

    model = landmark_to_BFM().to(device)
    ckpt = torch.load(os.path.join(opt.tgt_dir, 'landmark_to_BFM.pth'), map_location=device)
    model.load_state_dict(ckpt)
    os.makedirs(os.path.join(opt.src_dir, 'reenact_delta_from_mesh'), exist_ok=True)

    with torch.no_grad():
        model.eval()
        for i, data in enumerate(tqdm(test_dataloader)):
            normalized_mesh = data['normalized_mesh'].to(device).reshape(-1, 478*3)
            filename = data['filename'][0]
            pred = model(normalized_mesh)
            pred = pred[0].cpu()
            delta = pred.reshape(64, 1).clone()

            torch.save(delta, os.path.join(opt.src_dir, 'reenact_delta_from_mesh', filename))

            

