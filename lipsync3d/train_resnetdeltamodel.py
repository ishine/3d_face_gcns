import sys
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import torch
from torch.utils.data import DataLoader
from options import Options
from dataset import ResnetDeltaDataset
from loss import L2Loss, WeightedMSELoss, MediapipeLandmarkLoss
from audiodvp_utils.visualizer import Visualizer
from models import networks
import time
import os
from audiodvp_utils.util import create_dir
from tqdm import tqdm

if __name__ == '__main__':
    opt = Options().parse_args()
    device = opt.device
    
    train_dataset = ResnetDeltaDataset(opt)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,  # default not shuffle
        num_workers=opt.num_workers,
        drop_last=True  # the batch size cannot change during the training so the last uncomplete batch need to be dropped
    )
    
    valid_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,  # default not shuffle
        num_workers=opt.num_workers,
        drop_last=True  # the batch size cannot change during the training so the last uncomplete batch need to be dropped
    )
    
    visualizer = Visualizer(opt)
    model = networks.ResnetDeltaModel(opt).to(device)
    optimizer = torch.optim.Adam([{'params': model.fc.parameters()}, {'params': model.pretrained_model.parameters()}], lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_step, gamma=opt.lr_decay)
    criterionMedia = WeightedMSELoss(opt)
    criterionDelta = networks.CoefficientRegularization(use_mean=True)

    total_iters = 0
    for epoch in range(opt.num_epoch):
        epoch_start_time = time.time()
        epoch_iter = 0

        # --------- train ----------
        for i, data in enumerate(train_dataloader):
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            optimizer.zero_grad()
            input = data['input'].to(device)
            reference_geometry = data['reference_geometry'].to(device)
            mediapipe_mesh_gt = data['mediapipe_mesh_gt'].to(device)
            rotation = data['rotation'].to(device)
            translation = data['translation'].to(device)

            delta_pred, mediapipe_mesh_pred = model(input, reference_geometry, rotation, translation)
            
            mediapipeLoss = criterionMedia(mediapipe_mesh_pred, mediapipe_mesh_gt)
            deltaLoss = criterionDelta(delta_pred)
            loss = opt.lambda_land * mediapipeLoss + opt.lambda_reg * deltaLoss

            loss.backward()
            optimizer.step()

            if total_iters % opt.print_freq == 0:    # print training losses
                losses = {'mediapipeLoss': mediapipeLoss, 'deltaLoss': deltaLoss}
                visualizer.print_current_losses(epoch, epoch_iter, losses, 0, 0)
                visualizer.plot_current_losses(total_iters, losses)

        scheduler.step()
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.num_epoch, time.time() - epoch_start_time))

    
    create_dir(os.path.join(opt.tgt_dir, 'new_delta'))
    with tqdm(total=len(train_dataset)) as progress_bar:
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(valid_dataloader):
                input = data['input'].expand(opt.batch_size, -1, -1, -1).to(device)
                reference_geometry = data['reference_geometry'].expand(opt.batch_size, -1, -1).to(device)
                rotation = data['rotation'].expand(opt.batch_size, -1, -1).to(device)
                translation = data['translation'].expand(opt.batch_size, -1, -1).to(device)
                image_name = data['image_name']
                
                delta_pred, _ = model(input, reference_geometry, rotation, translation)
                torch.save(delta_pred[0].detach().cpu(), os.path.join(opt.tgt_dir, 'new_delta', image_name[0][:-4]+'.pt'))
                progress_bar.update(1)
            

