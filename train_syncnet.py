from os.path import join
from tqdm import tqdm

from models.syncnet import SyncNet
from audiodvp_utils import audio

import torch
from torch import nn
from torch import optim
from torch.utils import data as data_utils
from options.options import Options

import os, random
from audiodvp_utils.util import load_coef


opt = Options().parse_args()
global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = opt.syncnet_T
syncnet_mel_step_size = opt.syncnet_mel_step_size

class Dataset(object):
    def __init__(self, split):
        self.all_deltas = load_coef(os.path.join(opt.data_dir, 'delta'))
        split_idx = int(len(self.all_deltas) * 0.8)
        self.split = split
        self.offset = 0

        # if split == 'train':
        #     self.all_deltas = self.all_deltas[:split_idx]
            
        # else:
        #     self.all_deltas = self.all_deltas[split_idx:]
        #     self.offset = split_idx

        wavpath = join(opt.data_dir, 'audio', 'audio.aac')
        wav = audio.load_wav(wavpath, opt.sample_rate)

        self.orig_mel = audio.melspectrogram(wav).T

    def get_window(self, start_frame):
        window = self.all_deltas[start_frame: start_frame + syncnet_T]
        return window

    def crop_audio_window(self, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_idx = int(80. * ((start_frame + self.offset) / float(opt.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return self.orig_mel[start_idx : end_idx, :].copy()


    def __len__(self):
        return len(self.all_deltas) - syncnet_T

    def __getitem__(self, idx):
        wrong_idx = random.randint(0, len(self.all_deltas) - syncnet_T)

        if (idx == wrong_idx) or random.choice([True, False]):
            y = torch.ones(1).float()
            chosen = idx
        else:
            y = torch.zeros(1).float()
            chosen = wrong_idx

        window = self.get_window(chosen)

        mel = self.crop_audio_window(idx)

        # 64 x T
        x = torch.cat(window, dim=1)
        x = x.permute(1, 0) # T x 64
        mel = torch.FloatTensor(mel.T).unsqueeze(0)
        return x, mel, y

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
    prog_bar = tqdm(range(nepochs))
    for epoch in prog_bar:
        running_loss = 0.

        for step, (x, mel, y) in enumerate(train_data_loader):
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)
            loss = cosine_loss(a, v, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step % opt.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir, epoch)

            prog_bar.set_description('Epoch: {}, Loss: {}'.format(epoch, running_loss / (step + 1)))

        global_epoch += 1

def eval_model(test_data_loader, global_step, device, model, checkpoint_dir, epoch):
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):
            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)
            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

        averaged_loss = sum(losses) / len(losses)
        print('Epoch: {}, Eval average loss: {}'.format(epoch, averaged_loss))

        return

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if opt.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    checkpoint_dir = join(opt.data_dir, 'syncnet_ckpt')
    checkpoint_path = None

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=opt.syncnet_batch_size, shuffle=True,
        num_workers=opt.syncnet_num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=opt.syncnet_batch_size, shuffle=False,
        num_workers=opt.syncnet_num_workers)

    device = opt.device

    # Model
    model = SyncNet(opt).to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=opt.syncnet_lr)

    print('train set length: ', len(train_dataset))
    print('test set length: ', len(test_dataset))

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=opt.syncnet_checkpoint_interval,
          nepochs=opt.nepochs)