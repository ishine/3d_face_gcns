import os
import torch
import random
from audiodvp_utils import util, audio
from datasets.base_dataset import BaseDataset


class AudioDeltaDataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.data_dir = opt.data_dir

        if opt.isTrain:
            self.delta_list = util.load_coef(os.path.join(self.data_dir, 'delta'))[2:]

        wavpath = os.path.join(opt.data_dir, 'audio', 'audio.aac')
        wav = audio.load_wav(wavpath, opt.sample_rate)

        mel_idx_multiplier = 80./ opt.fps
        mel_step_size = opt.syncnet_mel_step_size
        mel = audio.melspectrogram(wav)
        mel_chunks = []
        i = 0

        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break

            mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
            i+=1
        
        print('Length of mel chunks: {}'.format(len(mel_chunks)))

        if opt.isTrain:
            min_len = min(len(self.delta_list), len(mel_chunks))
            self.delta_list = self.delta_list[:min_len]
            mel_chunks = mel_chunks[:min_len]
        else:
            first_two_frames = mel_chunks[:2]
            last_two_frames = mel_chunks[-2:]
            mel_chunks = first_two_frames + mel_chunks + last_two_frames  # add first and last two mel chunks for align purpose.
        self.mel_chunks = mel_chunks


    def __len__(self):
        return len(self.mel_chunks)

    def __getitem__(self, idx):
        if not self.opt.isTrain:
            filename = '%05d.pt' % idx
            mel = torch.FloatTensor(self.mel_chunks[idx]).unsqueeze(0)
            return {'mel' : mel, 'filename': filename}
        else:
            if idx < 2 or idx >= (len(self.mel_chunks) -2):
                idx = random.randint(2, len(self.mel_chunks)-3)

            mel = torch.FloatTensor(self.mel_chunks[idx]).unsqueeze(0)
            indiv_mels = torch.FloatTensor(self.mel_chunks[idx - 2:idx + 3]).unsqueeze(1)
            delta_gt = torch.stack(self.delta_list[idx - 2: idx + 3], dim=0).squeeze()
            return {'mel': mel, 'indiv_mels': indiv_mels, 'delta_gt': delta_gt}