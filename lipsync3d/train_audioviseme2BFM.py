import sys
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import torch
from torch.utils.data import DataLoader
from options import Options
from dataset import Audio2GeometryDataset, Audio2GeometrywithEnergyPitchDataset
from model import Audio2GeometryModel, Audio2GeometrywithEnergyPitchModel
from loss import L2Loss, WeightedMSELoss, TemporalLoss, emotionLoss
from audiodvp_utils.visualizer import Visualizer
from models import networks
import time
import os

if __name__ == '__main__':
    opt = Options().parse_args()
    device = opt.device
    
    train_dataset = Audio2GeometrywithEnergyPitchDataset(opt) 
    valid_dataset = Audio2GeometrywithEnergyPitchDataset(opt, is_valid=True) 

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,  # default not shuffle
        num_workers=opt.num_workers,
        drop_last=True  # the batch size cannot change during the training so the last uncomplete batch need to be dropped
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=opt.serial_batches,  # default not shuffle
        num_workers=opt.num_workers,
        drop_last=True  # the batch size cannot change during the training so the last uncomplete batch need to be dropped
    )
    
    visualizer = Visualizer(opt)
    model = Audio2GeometrywithEnergyPitchModel(opt, len(train_dataset)).to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_step, gamma=opt.lr_decay)
    criterionGeo = WeightedMSELoss(opt)
    criterionTemp = TemporalLoss()
    criterionEmotion = emotionLoss()

    total_iters = 0
    for epoch in range(opt.num_epoch):
        epoch_start_time = time.time()
        epoch_iter = 0

        # --------- train ----------
        for i, data in enumerate(train_dataloader):
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            optimizer.zero_grad()

            audio_feature = data['audio_feature'].to(device)
            audio_feature_prv = data['audio_feature_prv'].to(device)
            audio_feature_nxt = data['audio_feature_nxt'].to(device)
            target_exp_diff = data['target_exp_diff'].to(device)
            target_exp_diff_prv = data['target_exp_diff_prv'].to(device)
            target_exp_diff_nxt = data['target_exp_diff_nxt'].to(device)
            audio_energy = data['audio_energy'].to(device)
            audio_energy_prv = data['audio_energy_prv'].to(device)
            audio_energy_nxt = data['audio_energy_nxt'].to(device)
            audio_pitch = data['audio_pitch'].to(device)
            audio_pitch_prv = data['audio_pitch_prv'].to(device)
            audio_pitch_nxt = data['audio_pitch_nxt'].to(device)
            viseme = data['viseme'].to(device)
            viseme_prv = data['viseme_prv'].to(device)
            viseme_nxt = data['viseme_nxt'].to(device)
            
            index = data['index'].to(device)
            index_prv = data['index_prv'].to(device)
            index_nxt = data['index_nxt'].to(device)

            pred_exp_diff = model(audio_feature, viseme, audio_energy, audio_pitch, index) # model(audio_feature, index)
            pred_exp_diff_prv = model(audio_feature_prv, viseme_prv, audio_energy_prv, audio_pitch_prv, index_prv) # model(audio_feature_prv, index_prv)
            pred_exp_diff_nxt = model(audio_feature_nxt, viseme_nxt, audio_energy_nxt, audio_pitch_nxt, index_nxt) # model(audio_feature_nxt, index_nxt)
            
            geometryLoss = criterionGeo(target_exp_diff, pred_exp_diff)
            
            temporalLoss = criterionTemp(pred_exp_diff, pred_exp_diff_prv, pred_exp_diff_nxt, target_exp_diff, target_exp_diff_prv, target_exp_diff_nxt)
            EmotionLoss = criterionEmotion(model.mood, index, index_prv, index_nxt)
            loss = geometryLoss + opt.lambda_temporal * temporalLoss + opt.lambda_emotion * EmotionLoss

            loss.backward()
            optimizer.step()

            if total_iters % opt.print_freq == 0:    # print training losses
                losses = {'geometryLoss': geometryLoss, 'TemporalLoss': temporalLoss, 'EmotionLoss': EmotionLoss}
                visualizer.print_current_losses(epoch, epoch_iter, losses, 0, 0)
                visualizer.plot_current_losses(total_iters, losses)

        # --------- validate ----------
        model.eval()
        avg_geometryLoss, avg_mediapipeLoss = 0., 0.
        with torch.no_grad():
            for i, data in enumerate(valid_dataloader):
                audio_feature = data['audio_feature'].to(device)
                target_exp_diff = data['target_exp_diff'].to(device)
                audio_energy = data['audio_energy'].to(device)
                audio_pitch = data['audio_energy'].to(device)
                viseme = data['viseme'].to(device)

                pred_exp_diff = model(audio_feature, viseme, audio_energy, audio_pitch) #model(audio_feature)
                
                geometryLoss = criterionGeo(target_exp_diff, pred_exp_diff)

                avg_geometryLoss += geometryLoss * opt.batch_size / len(valid_dataset)

            losses = {'geometryLoss': avg_geometryLoss}
            print('Valid Results')
            visualizer.print_current_losses(epoch, epoch_iter, losses, 0, 0)

        scheduler.step()
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.num_epoch, time.time() - epoch_start_time))

    torch.save(model.cpu().state_dict(), os.path.join(opt.tgt_dir, 'audio_to_geometry_{}.pth'.format(opt.suffix)))
            

