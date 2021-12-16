
import tqdm
import torch

from options import Options
from dataset import Lipsync3DDataset

class PhonemeMeshModel(object):
    def __init__(self):
        self.memory = {}
        self.seen_pho = []
        self.mean_diff = None

    def remember(self, pho, geo_diff):
        if not pho in self.seen_pho:
            self.seen_pho.append(pho)
            self.memory[pho] = []
        self.memory[pho].append(geo_diff)

    def reduce_mean_diff(self):
        self.mean_diff = {}
        for pho in self.seen_pho:
            all_diff = torch.stack(self.memory[pho], dim=0)
            self.mean_diff[pho] = all_diff.mean(dim=0)

    def recall(self, pho):
        if not pho in self.seen_pho:
            return torch.zeros(478, 3)
        return self.mean_diff[pho]

    def save(self, path):
        state_dict = {
            'mem': self.memory,
            'seen_pho': self.seen_pho,
            'mean_diff': self.mean_diff
        }
        torch.save(state_dict, path)
    
    def load(self, path):
        sd = torch.load(path)
        self.memory = sd['mem']
        self.seen_pho = sd['seen_pho']
        self.mean_diff = sd['mean_diff']


def test1():
    print("phoneme mesh model segment mean")
    opt = Options().parse_args()
    dataset = Lipsync3DDataset(opt)
    print(len(dataset))
    model = PhonemeMeshModel()
    save_path = 'phoneme_mesh_model_segment.pt'
    print("remembering dataset")
    dsitem = dataset[0]
    ref_mesh = dsitem['reference_mesh'] # [478, 3]
    for dsitem in tqdm.tqdm(dataset):
        src_seq = dsitem['src_seq'] # [24]
        pho = src_seq[12].item()
        norm_mesh = dsitem['normalized_mesh'] # [478, 3]
        geo_diff = norm_mesh - ref_mesh
        model.remember(pho, geo_diff)
    print(len(model.seen_pho))
    model.reduce_mean_diff()
    model.save(save_path)


def test2():
    print("phoneme mesh model middle")
    opt = Options().parse_args()
    dataset = Lipsync3DDataset(opt)
    model = PhonemeMeshModel()
    save_path = 'phoneme_mesh_model_middle.pt'
    dsitem = dataset[0]
    ref_mesh = dsitem['reference_mesh'] # [478, 3]
    middle_indices = []
    last_pho = 0
    pho_len = 0
    st_idx = -1
    print("first pass")
    for idx, dsitem in enumerate(tqdm.tqdm(dataset)):
        src_seq = dsitem['src_seq'] # [24]
        pho = src_seq[12].item()
        if pho == last_pho:
            pho_len += 1
        else:
            last_mid_idx = st_idx + pho_len//2
            middle_indices.append(last_mid_idx)
            last_pho = pho
            pho_len = 1
            st_idx = idx
    last_mid_idx = st_idx + pho_len//2
    middle_indices.append(last_mid_idx)
    print("second pass")
    for mid_idx in tqdm.tqdm(middle_indices):
        dsitem = dataset[mid_idx]
        src_seq = dsitem['src_seq'] # [24]
        pho = src_seq[12].item()
        norm_mesh = dsitem['normalized_mesh'] # [478, 3]
        geo_diff = norm_mesh - ref_mesh
        model.remember(pho, geo_diff)
    model.reduce_mean_diff()
    model.save(save_path)


if __name__ == '__main__':
    test2()


