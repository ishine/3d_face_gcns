import sys
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import torch
from torch.utils.data import DataLoader
from options import Options
from dataset import Audio2GeometryDataset
from model import Audio2GeometryModel
from loss import calculate_geoLoss
import time
import os
from tqdm import tqdm

if __name__ == '__main__':
    opt = Options().parse_args()
    device = opt.device
    dataset = Audio2GeometryDataset(opt)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,  # default not shuffle
        num_workers=opt.num_workers,
        drop_last=True  # the batch size cannot change during the training so the last uncomplete batch need to be dropped
    )

    model = Audio2GeometryModel(opt).to(device)
    ckpt = torch.load(os.path.join(opt.tgt_dir, 'audio_to_geometry_{}.pth'.format(opt.suffix)), map_location=device)
    model.load_state_dict(ckpt)
    os.makedirs(os.path.join(opt.src_dir, 'reenact_geometry'), exist_ok=True)

    with torch.no_grad():
        model.eval()
        for i, data in enumerate(tqdm(test_dataloader)):
            audio_feature = data['audio_feature'].to(device)
            filename = data['filename'][0]

            geometry_pred = model(audio_feature)
            geometry_pred = geometry_pred[0].cpu().clone()

            torch.save(geometry_pred, os.path.join(opt.src_dir, 'reenact_geometry', filename))

            

