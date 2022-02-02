import tgt
import os
from audiodvp_utils import util
import math
import torch
from tqdm import tqdm
import scipy.io as sio
import numpy as np


def get_metadata(src_dir, is_kor=True):
    metadata_path = os.path.join(src_dir, 'metadata.pt')
    
    # if os.path.exists(metadata_path):
    #     return torch.load(metadata_path)
    
    image_list = util.get_file_list(os.path.join(src_dir, 'crop'))
    num_frames = len(image_list)

    phoneme_list = [None for _ in range(num_frames)]

    total_time = 0
    file_path = os.path.join(src_dir, 'flist.txt')

    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            fname = line.split(' ')[-1][6:-5]
            tg_name = "{}.TextGrid".format(fname)
            tg_path = os.path.join(src_dir, "textgrid/{}".format(tg_name))   
            textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
            tier = textgrid.get_tier_by_name('phones')
            time_in_file = 0
            
            for t in tier._objects:
                s, e, p = t.start_time, t.end_time, t.text
                if int((total_time + time_in_file + e - s) * 25) < len(phoneme_list):
                    for i in range(math.ceil((total_time + time_in_file) * 25), int((total_time + time_in_file + e - s) * 25)+1):
                        phoneme_list[i] = p if p != '' else 'silent'
                else:
                    for i in range(math.ceil((total_time + time_in_file) * 25), len(phoneme_list)):
                        phoneme_list[i] = p if p != '' else 'silent'
                
                time_in_file += e - s
            
            total_time += int(time_in_file * 60) / 60
    print(total_time)
    pho_segs = {}
    seg_id = 0
    indices = []
    current_pho = None
    current_stress = None

    for idx in range(len(phoneme_list)):
        orig_pho = phoneme_list[idx]
        if is_kor or orig_pho[-1].isalpha():
            stress = None
            pho = orig_pho
        else:
            stress = int(orig_pho[-1])
            pho = orig_pho[:-1]
            
        if pho not in pho_segs:
            pho_segs[pho] = []
        
        if current_pho != pho:
            if current_pho == None:
                current_pho = pho
                current_stress = stress
                indices.append(idx)
            else:
                segment_item = {
                    "id": seg_id,
                    "stress": current_stress,
                    "indices": indices
                }
                pho_segs[current_pho].append(segment_item)
                current_pho = pho
                current_stress = stress
                seg_id +=1
                indices = []
                indices.append(idx)
        else:
            indices.append(idx)
    
    segment_item = {
        "id": seg_id,
        "stress" : current_stress,
        "indices": indices
    }
    pho_segs[current_pho].append(segment_item)
    
    
    n_segments = seg_id + 1
    metadata = {
        "pho_segs": pho_segs,
        "n_segments": n_segments
    }
    
    torch.save(metadata, metadata_path)
    print("metadata saved.")
    
    return metadata

def get_concatenation_cost(src_dir):
    cost_path = os.path.join(src_dir, 'concatenation_cost.pt')
    
    if os.path.exists(cost_path):
        return torch.load(cost_path)
    
    metadata = get_metadata(src_dir)
    n_segments = metadata["n_segments"]
    delta_list = util.load_coef(os.path.join(src_dir, 'delta'))
    
    matlab_data_path = 'renderer/data/data.mat'
    mat_data = sio.loadmat(matlab_data_path)
    exp_base = torch.from_numpy(mat_data['exp_base']).cuda()
        
    st_deltas = torch.zeros((n_segments, 64)).cuda()
    ed_deltas = torch.zeros((n_segments, 64)).cuda()
    
    print("reading segments")
    for pho in tqdm(metadata["pho_segs"]):
        for seg in metadata["pho_segs"][pho]:
            seg_id = seg["id"]
            st_deltas[seg_id] = delta_list[seg["indices"][0]].squeeze().cuda()
            ed_deltas[seg_id] = delta_list[seg["indices"][-1]].squeeze().cuda()
            
    print("computing costs")
    costs = {
        "l1_err": [],
        "mse": []
    }
    st_deltas = st_deltas
    ed_deltas = ed_deltas
    for ed_idx in tqdm(range(n_segments)):
        ed_delta = ed_deltas[ed_idx].unsqueeze(0)
        diff = torch.matmul(exp_base, (st_deltas - ed_delta).permute(1, 0))
        l1_err = torch.mean(torch.abs(diff), dim=0)
        mse = torch.mean(torch.abs(diff) ** 2, dim=0)
        costs['l1_err'].append(l1_err.cpu())
        costs['mse'].append(mse.cpu())
    
    for k in costs:
        costs[k] = torch.stack(costs[k], dim=0).numpy()
        
    torch.save(costs, cost_path)

    return costs


class ViterbiDecoder():
    
    def __init__(self, n_ele, gen_eles, target_cost_fn, concat_cost_mat):
        # n_ele: int (number of gen_ele in bank)
        # gen_eles: n_ele-list of gen_ele
        # target_cost_fn: (gen_ele, target_ele) -> float
        # concat_cost_mat: (n_ele, n_ele) array
        self.n_ele = n_ele
        self.gen_eles = gen_eles
        self.target_cost_fn = target_cost_fn
        self.concat_cost_mat = concat_cost_mat

    def decode(self, target, debug=False):
        # target: T-list of target_ele
        # ---
        # result: T-list of int (optimal gen_ele indicies)

        print("preparing Viterbi")
        # prepare input
        T = len(target)
        K = len(self.gen_eles)
        A = self.concat_cost_mat # [K_prev, K_cur]
        By = []
        for target_ele in target:
            By_t = []
            for gen_ele in self.gen_eles:
                By_t.append(self.target_cost_fn(gen_ele, target_ele))
            By.append(By_t)
        By = np.array(By) # [T, K]

        # intiialize
        T1 = np.empty((K, T), 'd') # cost
        T2 = np.empty((K, T), 'i') # path
        T1[:, 0] = By[0, :]
        T2[:, 0] = 0

        print("running Viterbi")
        for i in range(1, T):
            T1[:, i] = np.min(T1[np.newaxis, :, i-1] + A.T + By[i, :, np.newaxis], axis=1)
            # [K_cur] <-min- [K_cur, K_prev] <- [1, K_prev] + [K_cur, K_prev] + [K_cur, 1]
            T2[:, i] = np.argmin(T1[np.newaxis, :, i-1] + A.T, axis=1)

        x = np.empty(T, 'i')
        x[-1] = np.argmin(T1[:, T-1])
        for i in reversed(range(1, T)):
            x[i-1] = T2[x[i], i]

        result = x
        print('viterbi done!')

        return result
    

class TargetElement(object):
    def __init__(self, pho, stress, duration):
        self.pho = pho
        self.stress = stress
        self.duration = duration

class GenElement(object):
    def __init__(self, pho, stress, duration, seg_info):
        self.pho = pho
        self.stress = stress
        self.duration = duration
        self.seg_info = seg_info

def textgrid2targetseq(tg_path, is_kor=True):
    target_seq = []
    total_time = 0
  
    textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
    tier = textgrid.get_tier_by_name('phones')
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text
        orig_pho = p if p != '' else 'silent'
        
        if is_kor or orig_pho[-1].isalpha():
            pho = orig_pho
            stress = None
        else:
            pho = orig_pho[:-1]
            stress = int(orig_pho[-1])
        duration = int((total_time + e - s)* 25) - int((total_time * 25))
        if duration != 0:
            target_seq.append(TargetElement(pho, stress, duration))
                
        total_time += e - s
    
    return target_seq


def viterbi_algorithm(src_dir, tg_path):
    
    target = textgrid2targetseq(tg_path)
    metadata = get_metadata(src_dir)
    gen_eles = [None for _ in range(metadata["n_segments"])]
    cnt = 0
    for pho in metadata["pho_segs"]:
        for seg in metadata["pho_segs"][pho]:
            cnt += 1
            gen_eles[seg["id"]] = GenElement(pho, seg["stress"], len(seg["indices"]), seg)
    assert None not in gen_eles
    
    phos = set([gen_ele.pho for gen_ele in gen_eles])
    
    # define target cost
    def target_cost_fn(gen_ele, target_ele):
        cost = 0.0
        if gen_ele.pho != target_ele.pho:
            return 100.0
        if gen_ele.stress != target_ele.stress:
            cost += 0.1

        dur_weight = 1.0
        cost += dur_weight * abs(math.log(gen_ele.duration/target_ele.duration)) # < 0.7 for double length
        return cost
    
    costs = get_concatenation_cost(src_dir)
    concat_cost_mat = 100 * costs["l1_err"]
    
    viterbi = ViterbiDecoder(metadata["n_segments"], gen_eles, target_cost_fn, concat_cost_mat)
    result_idx = viterbi.decode(target)
    
    gen_seq = [gen_eles[idx] for idx in result_idx]
    
    verbose = True
    if verbose:
        last_gen_idx = None
        for target_ele, gen_ele, gen_idx in zip(target, gen_seq, result_idx):
            if last_gen_idx is None:
                concat_cost = None
            else:
                concat_cost = concat_cost_mat[last_gen_idx, gen_idx]
                concat_cost = "{:.3f}".format(concat_cost)
            last_gen_idx = gen_idx
            print(f"|{target_ele.pho} {gen_ele.pho} \t| {target_ele.duration} {gen_ele.duration} \t| {concat_cost} \t|{gen_idx}")

    full_delta_list = util.load_coef(os.path.join(src_dir, 'delta'))
    
    delta_list = []
    crop_lip_indices = []
    for target_ele, gen_ele in zip(target, gen_seq):
        for i in range(target_ele.duration):
            if target_ele.pho == 'silent':
                delta_list.append(torch.zeros((64, 1)))
                crop_lip_indices.append(0)            
            elif (i == (target_ele.duration // 2)):
                delta_list.append(full_delta_list[gen_ele.seg_info["indices"][gen_ele.duration // 2]])
                crop_lip_indices.append(gen_ele.seg_info["indices"][gen_ele.duration // 2])
            else:
                delta_list.append(None)
                crop_lip_indices.append(gen_ele.seg_info["indices"][gen_ele.duration // 2])
                
    interpolated_delta_list = delta_interpolation(delta_list)
    return interpolated_delta_list, crop_lip_indices


def delta_interpolation(delta_list):
    interpolated_delta_list = []

    prev_delta = None
    prev_delta_idx = -1
    for idx in range(len(delta_list)):
        if delta_list[idx] == None:
            if idx == (len(delta_list) - 1):
                for _ in range(prev_delta_idx, idx):
                    interpolated_delta_list.append(prev_delta)
            else:
                continue
        
        else:
            if prev_delta == None:
                for _ in range(prev_delta_idx, idx):
                    interpolated_delta_list.append(delta_list[idx])
                prev_delta = delta_list[idx]
                prev_delta_idx = idx
            else:
                for i in range(idx - prev_delta_idx):
                    interpolated_delta = delta_list[prev_delta_idx] + (delta_list[idx] - delta_list[prev_delta_idx]) * (i+1) / (idx - prev_delta_idx)
                    interpolated_delta_list.append(interpolated_delta)
                prev_delta = delta_list[idx]
                prev_delta_idx = idx
    
    return interpolated_delta_list
