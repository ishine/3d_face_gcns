# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import data
import numpy as np
from util.util import masktorgb
from options.test_options import TestOptions
# from options.train_options import TrainOptions
from models.pix2pix_model import Pix2PixModel
from tqdm import tqdm

opt = TestOptions().parse()

torch.manual_seed(0)
dataloader = data.create_dataloader(opt)
dataloader.dataset[0]

model = Pix2PixModel(opt)
model.eval()

save_root = os.path.join(opt.dataroot, 'images')

if not os.path.exists(save_root):
    os.makedirs(save_root)
    
# test
for i, data_i in enumerate(tqdm(dataloader)):
    out = model(data_i, mode='inference')

    pre = out['fake_image'][0].data.cpu()
    label = data_i['label'][0][9:12, :, :].cpu()

    batch_size = pre.shape[0]

    pre = (pre + 1) / 2
    vutils.save_image(pre, os.path.join(save_root, '%05d_fake.png' % (i+1)),
            nrow=1, padding=0, normalize=False)

    label = (label + 1) / 2
    vutils.save_image(label, os.path.join(save_root, '%05d_real.png' % (i+1)),
                        nrow=1, padding=0, normalize=False)
