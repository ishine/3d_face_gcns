import sys
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import torch
from torch.utils.data import DataLoader
from options import Options
from dataset import Audio2GeometryDataset
from model import Audio2GeometryModel
from loss import L2Loss, WeightedMSELoss
from audiodvp_utils.visualizer import Visualizer
from models import networks
import time
import os
from adabound import AdaBound
from renderer import FaceModel

if __name__ == '__main__':
    opt = Options().parse_args()
    device = opt.device
    
    dataset = Audio2GeometryDataset(opt)
    train_dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,  # default not shuffle
        num_workers=opt.num_workers,
        drop_last=True  # the batch size cannot change during the training so the last uncomplete batch need to be dropped
    )

    visualizer = Visualizer(opt)
    facemodel = FaceModel(opt, batch_size=opt.batch_size)
    model = Audio2GeometryModel(opt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_step, gamma=opt.lr_decay)
    criterionGeo = WeightedMSELoss(opt)

    total_iters = 0
    for epoch in range(opt.num_epoch):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(train_dataloader):
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            optimizer.zero_grad()

            audio_feature = data['audio_feature'].to(device)
            geometry = data['geometry'].to(device)

            geometry_pred = model(audio_feature)

            geometryLoss = criterionGeo(geometry, geometry_pred)

            loss = geometryLoss

            loss.backward()
            optimizer.step()

            if total_iters % opt.print_freq == 0:    # print training losses
                losses = {'geometryLoss': geometryLoss}
                visualizer.print_current_losses(epoch, epoch_iter, losses, 0, 0)
                visualizer.plot_current_losses(total_iters, losses)

        scheduler.step()
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.num_epoch, time.time() - epoch_start_time))

    torch.save(model.cpu().state_dict(), os.path.join(opt.tgt_dir, 'audio_to_geometry.pth'))
            

