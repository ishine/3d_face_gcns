import sys
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import torch
from torch.utils.data import DataLoader
from options import Options
from dataset import Landmark2BFMDataset
from model import landmark_to_BFM
from loss import calculate_geoLoss
from audiodvp_utils.visualizer import Visualizer
from models import networks
import time
import os

if __name__ == '__main__':
    opt = Options().parse_args()
    device = opt.device

    dataset = Landmark2BFMDataset(opt)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,  # default not shuffle
        num_workers=opt.num_workers,
        drop_last=True  # the batch size cannot change during the training so the last uncomplete batch need to be dropped
    )

    visualizer = Visualizer(opt)

    model = landmark_to_BFM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_step, gamma=opt.lr_decay)

    criterionGeo = calculate_geoLoss(opt)
    regularizationAlpha = networks.CoefficientRegularization()
    regularizationDelta = networks.CoefficientRegularization()
    reference_alpha = torch.load(os.path.join(opt.tgt_dir, 'reference_alpha.pt')).expand(opt.batch_size, -1, -1).to(device)

    total_iters = 0
    for epoch in range(opt.num_epoch):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(train_dataloader):
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            optimizer.zero_grad()

            alpha = data['alpha'].to(device)
            delta = data['delta'].to(device)
            normalized_mesh = data['normalized_mesh'].to(device).reshape(-1, 478*3)

            pred = model(normalized_mesh)
            # alpha_pred = pred[:, :80].view(-1, 80, 1)
            delta_pred = pred.view(-1, 64, 1)

            # alphaLoss = regularizationAlpha(alpha_pred)
            deltaLoss = regularizationDelta(delta_pred)
            geoLoss = criterionGeo(alpha, delta, reference_alpha, delta_pred)
            loss = opt.lambda_land * geoLoss + opt.lambda_reg * (opt.lambda_delta * deltaLoss)

            loss.backward()
            optimizer.step()

            if total_iters % opt.print_freq == 0:    # print training losses
                losses = {'geoLoss': geoLoss,  'deltaLoss': deltaLoss}
                visualizer.print_current_losses(epoch, epoch_iter, losses, 0, 0)
                visualizer.plot_current_losses(total_iters, losses)
        
        scheduler.step()

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.num_epoch, time.time() - epoch_start_time))

    torch.save(model.cpu().state_dict(), os.path.join(opt.tgt_dir, 'landmark_to_BFM.pth'))
            

