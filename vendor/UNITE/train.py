# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import torch
import torchvision.utils as vutils
import sys
import cv2
from PIL import Image
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.util import print_current_errors
from trainers.pix2pix_trainer import Pix2PixTrainer
import torch.nn.functional as F

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

#torch.manual_seed(0)
# load the dataset
dataloader = data.create_dataloader(opt)
len_dataloader = len(dataloader)
dataloader.dataset[11]

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create trainer for our model
trainer = Pix2PixTrainer(opt, resume_epoch=iter_counter.first_epoch)

im_sv_path = os.path.join(opt.checkpoints_dir, 'images')
            
if not os.path.exists(im_sv_path):
    os.makedirs(im_sv_path)
                
# save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), opt.name)
for epoch in iter_counter.training_epochs():
    opt.epoch = epoch
    
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        # Training

        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)
        trainer.run_discriminator_one_step(data_i)
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)

        if iter_counter.needs_displaying():
            imgs_num = data_i['label'].shape[0]
            
            label = data_i['label'][:, 9:12, :, :]

            imgs = torch.cat((label.cpu(), trainer.out['weight1'].cpu()*255, trainer.out['weight2'].cpu()*255,
                                data_i['ref'].cpu(), trainer.out['warp_tmp'].cpu(),
                                trainer.get_latest_generated().data.cpu(), data_i['image'].cpu()), 0)
            
            vutils.save_image(imgs, im_sv_path + '/' + str(epoch) + '_' + str(iter_counter.total_steps_so_far) + '.png',
                        nrow=imgs_num, padding=0, normalize=True)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
