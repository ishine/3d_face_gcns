import os
import torch

from audiodvp_utils import util
from datasets.base_dataset import BaseDataset


class Audio2ExpressionDataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.data_dir = opt.data_dir
        self.seq_len = opt.seq_len

        self.feature_list = util.load_coef(os.path.join(self.data_dir, 'feature'))
        self.filenames = util.get_file_list(os.path.join(self.data_dir, 'feature'))

        if opt.isTrain:
            self.delta_list = util.load_coef(os.path.join(self.data_dir, 'delta'))
            minlen = min(len(self.feature_list), len(self.delta_list))
            train_idx = int(minlen * 4 / 5)
            self.feature_list = self.feature_list[:train_idx]
            self.filenames = self.filenames[:train_idx]
        
        print('audio2expr datset length : ', len(self.feature_list))

    def __len__(self):
        return len(self.feature_list)

    def getitem(self, index):
        cur_feat = self.feature_list[index].unsqueeze(0).float()

        r = self.seq_len // 2
        for i in range(1, r):
            index_seq = index - i
            if index_seq < 0: index_seq = 0

            prv_feat = self.feature_list[index_seq].unsqueeze(0).float()
            cur_feat = torch.cat([prv_feat, cur_feat], 0)
        
        for i in range(1, self.seq_len - r + 1):
            index_seq = index + i
            max_idx = len(self.feature_list) - 1
            if index_seq > max_idx: index_seq = max_idx

            nxt_feat = self.feature_list[index_seq].unsqueeze(0).float()
            cur_feat = torch.cat([cur_feat, nxt_feat], 0)
        
        return cur_feat


    def __getitem__(self, index):
        filename = os.path.basename(self.filenames[index])
        feature = self.getitem(index)

        if not self.opt.isTrain:
            return {'feature': feature, 'filename': filename}
        else:
            feature_prv = self.getitem(max(index - 1, 0))
            feature_nxt = self.getitem(min(index + 1, len(self.feature_list) - 1))

            delta = self.delta_list[index]
            delta_prv = self.delta_list[max(index - 1, 0)]
            delta_nxt = self.delta_list[min(index + 1, len(self.feature_list) - 1)]

            return {
                'feature': feature,
                'feature_prv': feature_prv,
                'feature_nxt': feature_nxt,
                'filename': filename, 
                'expressions': delta,
                'expressions_prv': delta_prv,
                'expressions_nxt': delta_nxt
            }
