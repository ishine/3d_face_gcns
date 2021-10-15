import sys
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import torch
from torch.utils.data import DataLoader
from options import Options
from dataset import Lipsync3DDataset
from model import Lipsync3DModel
from loss import L2Loss, L1Loss
from audiodvp_utils.visualizer import Visualizer
import time
import os

if __name__ == '__main__':
    opt = Options().parse_args()
    device = opt.device

    dataset = Lipsync3DDataset(opt)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,  # default not shuffle
        num_workers=opt.num_workers,
        drop_last=True  # the batch size cannot change during the training so the last uncomplete batch need to be dropped
    )

    visualizer = Visualizer(opt)

    model = Lipsync3DModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    criterionGeo = L2Loss()
    criterionTex = L1Loss()

    total_iters = 0
    for epoch in range(opt.num_epoch):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(train_dataloader):
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            optimizer.zero_grad()

            audio_feature = data['audio_feature'].to(device)
            reference_mesh = data['reference_mesh'].to(device)
            normalized_mesh = data['normalized_mesh'].to(device)

            # geometry_diff, texture_dfif = model(audio_feature)
            geometry_diff = model(audio_feature)
            geometry_diff = geometry_diff.reshape(-1, 478, 3)
            geometry = reference_mesh + geometry_diff
            geoLoss = criterionGeo(geometry, normalized_mesh)
            loss = opt.lambda_geo * geoLoss

            loss.backward()
            optimizer.step()

            if total_iters % opt.print_freq == 0:    # print training losses
                losses = {'geoLoss': geoLoss}
                visualizer.print_current_losses(epoch, epoch_iter, losses, 0, 0)
                visualizer.plot_current_losses(total_iters, losses)
            

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.num_epoch, time.time() - epoch_start_time))

    torch.save(model.cpu().state_dict(), os.path.join(opt.tgt_dir, 'Lipsync3dnet.pth'))
            

