import sys

from cv2 import normalize
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import os
import torch
import numpy as np
import librosa
from utils import landmarkdict_to_normalized_mesh_tensor, landmarkdict_to_mesh_tensor, get_downsamp_trans, \
    load_mediapipe_mesh_dict, get_mel_spectogram, get_audio_stft, get_audio_energy_pitch, get_viseme_list
from audiodvp_utils import util
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
from lib.mesh_io import read_obj
from gcn_util.utils import init_sampling
import scipy.io as sio
from torchvision import transforms
from PIL import Image

class Lipsync3DDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.src_dir = opt.src_dir
        self.tgt_dir = opt.tgt_dir

        stft_path = os.path.join(self.src_dir, 'audio/audio_stft.pt')
        if not os.path.exists(stft_path):
            audio = librosa.load(os.path.join(self.src_dir, 'audio/audio.wav'),16000)[0]
            audio_stft = librosa.stft(audio, n_fft=510, hop_length=160, win_length=480)
            self.audio_stft = torch.from_numpy(np.stack((audio_stft.real, audio_stft.imag)))
            torch.save(self.audio_stft, os.path.join(self.src_dir, 'audio/audio_stft.pt'))
        else:
            self.audio_stft = torch.load(os.path.join(self.src_dir, 'audio/audio_stft.pt'))
        
        self.mesh_dict_list = util.load_coef(os.path.join(self.tgt_dir, 'mesh_dict'))
        self.filenames = util.get_file_list(os.path.join(self.tgt_dir, 'mesh_dict'))
        reference_mesh_dict = torch.load(os.path.join(self.tgt_dir, 'reference_mesh.pt'))
        self.reference_mesh = landmarkdict_to_normalized_mesh_tensor(reference_mesh_dict)

        if opt.isTrain:
            minlen = min(len(self.mesh_dict_list), self.audio_stft.shape[2] // 4)
            train_idx = int(minlen * self.opt.train_rate)
            self.mesh_dict_list = self.mesh_dict_list[:train_idx]
            self.filenames = self.filenames[:train_idx]
        
        print('training set size: ', len(self.filenames))


    def __len__(self):
        return min(self.audio_stft.shape[2] // 4, len(self.filenames))

    def __getitem__(self, index):
        audio_idx = index * 4

        audio_feature_list = []
        for i in range(audio_idx - 12, audio_idx + 12):
            if i < 0:
                audio_feature_list.append(self.audio_stft[:, :, 0])
            elif i >= self.audio_stft.shape[2]:
                audio_feature_list.append(self.audio_stft[:, :, -1])
            else:
                audio_feature_list.append(self.audio_stft[:, :, i])

        audio_feature = torch.stack(audio_feature_list, 2)

        filename = os.path.basename(self.filenames[index])

        if not self.opt.isTrain:
            landmark_dict = self.mesh_dict_list[index]
            normalized_mesh = landmarkdict_to_normalized_mesh_tensor(landmark_dict)
            R = torch.from_numpy(landmark_dict['R']).float()
            t = torch.from_numpy(landmark_dict['t']).float()
            c = float(landmark_dict['c'])

            return {'audio_feature': audio_feature, 'filename': filename, 
                    'reference_mesh': self.reference_mesh, 'normalized_mesh': normalized_mesh,
                    'R': R, 't': t, 'c': c}
        else:
            landmark_dict = self.mesh_dict_list[index]
            normalized_mesh = landmarkdict_to_normalized_mesh_tensor(landmark_dict)
            
            return {
                'audio_feature': audio_feature, 'filename': filename,
                'reference_mesh' : self.reference_mesh, 'normalized_mesh': normalized_mesh 
            }


class Landmark2BFMDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.mesh_dir = opt.mesh_dir
        self.mesh_dict_list = util.load_coef(self.mesh_dir)
        self.filenames = util.get_file_list(self.mesh_dir)
        self.mesh_mean, self.mesh_std = self.load_mesh_statistic()

        if opt.isTrain:
            self.bfm_dir = opt.bfm_dir
            self.alpha_list = util.load_coef(os.path.join(self.bfm_dir, 'alpha'))
            self.delta_list = util.load_coef(os.path.join(self.bfm_dir, 'delta'))
            self.beta_list = util.load_coef(os.path.join(self.bfm_dir, 'beta'))
            self.gamma_list = util.load_coef(os.path.join(self.bfm_dir, 'gamma'))
            self.angle_list = util.load_coef(os.path.join(self.bfm_dir, 'rotation'))
            self.translation_list = util.load_coef(os.path.join(self.bfm_dir, 'translation'))
            self.face_emb_list = util.load_face_emb(self.bfm_dir)

            self.image_list = util.get_file_list(os.path.join(self.opt.bfm_dir, 'crop'))
            self.face_landmark_dict = self.load_face_landmark_dict()
            

            minlen = len(self.mesh_dict_list)
            train_idx = int(minlen * self.opt.train_rate)
            self.mesh_dict_list = self.mesh_dict_list[:train_idx]
            self.filenames = self.filenames[:train_idx]
        
        print('training set size: ', len(self.filenames))


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):

        filename = os.path.basename(self.filenames[index])

        if not self.opt.isTrain:
            landmark_dict = self.mesh_dict_list[index]
            normalized_mesh = landmarkdict_to_normalized_mesh_tensor(landmark_dict)
            normalized_mesh = (normalized_mesh - self.mesh_mean) / self.mesh_std

            return {'filename': filename, 
                    'normalized_mesh': normalized_mesh}
        else:
            landmark_dict = self.mesh_dict_list[index]
            normalized_mesh = landmarkdict_to_normalized_mesh_tensor(landmark_dict)
            normalized_mesh = (normalized_mesh - self.mesh_mean) / self.mesh_std

            alpha = self.alpha_list[index]
            delta = self.delta_list[index]
            beta = self.beta_list[index]
            face_emb = self.face_emb_list[index]
            gamma = self.gamma_list[index]
            angle = self.angle_list[index]
            translation = self.translation_list[index]

            image_name = self.image_list[index]
            face_landmark_gt = torch.tensor(self.face_landmark_dict[image_name])

            return {
                'filename': filename,
                'normalized_mesh': normalized_mesh,
                'alpha': alpha,
                'delta': delta,
                'beta': beta,
                'gamma': gamma,
                'angle': angle,
                'translation': translation,
                'face_emb': face_emb,
                'face_landmark_gt': face_landmark_gt
            }

    def load_face_landmark_dict(self):
        landmark_path = os.path.join(self.opt.bfm_dir, 'landmark.pkl')

        if not os.path.exists(landmark_path):
            util.landmark_detection(self.image_list, landmark_path)

        with open(landmark_path, 'rb') as f:
            landmark_dict = pickle.load(f)

        return landmark_dict

    def load_mesh_statistic(self):
        mesh_statistic_path = os.path.join(self.opt.tgt_dir, 'mesh_statistic.pkl')

        if not os.path.exists(mesh_statistic_path):
            self.calculate_mesh_statistic(self.opt.mesh_dir, mesh_statistic_path)
        
        with open(mesh_statistic_path, 'rb') as f:
            mesh_statistic_dict = pickle.load(f)

        return mesh_statistic_dict['mean'], mesh_statistic_dict['std']
    
    def calculate_mesh_statistic(self, mesh_dir, save_path):
        mesh_dict_list = util.load_coef(mesh_dir)

        mesh_list = []
        for mesh_dict in tqdm(mesh_dict_list):
            mesh_list.append(landmarkdict_to_normalized_mesh_tensor(mesh_dict))

        total_mesh = torch.stack(mesh_list)
        total_mesh = total_mesh.reshape(-1, 3)
        mean = total_mesh.mean(dim=0).reshape(1, 3)
        std = total_mesh.std(dim=0).reshape(1, 3)

        mesh_statistic = {'mean': mean, 'std': std}

        with open(save_path, 'wb') as f:
            pickle.dump(mesh_statistic, f)

        return


class Audio2GeometryDataset(Dataset):
    def __init__(self, opt, is_valid=False):
        self.opt = opt
        self.src_dir = opt.src_dir
        self.tgt_dir = opt.tgt_dir
        self.device = opt.device

        self.audio_feature = get_audio_stft(self.src_dir)

        self.filenames = util.get_file_list(os.path.join(self.tgt_dir, 'alpha'))
        self.start_idx = 0
        self.len = len(self.audio_feature)

        if opt.isTrain:
            self.downsamp_trans = get_downsamp_trans()

            self.delta_list = util.load_coef(os.path.join(self.tgt_dir, 'delta'))
            self.id_base, self.exp_base, self.geo_mean = self.mat_datum()

            self.image_list = util.get_file_list(os.path.join(self.opt.tgt_dir, 'crop'))
            minlen = min(len(self.image_list), len(self.audio_feature))
            self.len = int(minlen * self.opt.train_rate)
            if is_valid:
                self.start_idx = self.len
                self.len = minlen - self.len

        print('dataset size: ', len(self))

    def mat_datum(self):
        mat_data = sio.loadmat(self.opt.matlab_data_path)

        exp_base = torch.from_numpy(mat_data['exp_base']).reshape(-1, 3 * 64)
        exp_base = torch.mm(self.downsamp_trans, exp_base).reshape(-1, 64)

        id_base = torch.from_numpy(mat_data['id_base']).reshape(-1, 3 * 80)
        id_base = torch.mm(self.downsamp_trans, id_base).reshape(-1, 80)

        geo_mean = torch.from_numpy(mat_data['geo_mean']).reshape(-1, 3)
        geo_mean = torch.mm(self.downsamp_trans, geo_mean).reshape(-1, 1)

        return id_base, exp_base, geo_mean

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = index + self.start_idx
        audio_feature = self.audio_feature[index]

        filename = '%05d' % (index + 1) + '.pt'

        if not self.opt.isTrain:
            return {'audio_feature': audio_feature, 'filename': filename}
        else:
            audio_feature_prv = self.audio_feature[max(index - 1, 0)]
            audio_feature_nxt = self.audio_feature[min(index + 1, len(self) - 1)]
            
            delta = self.delta_list[index]
            delta_prv = self.delta_list[max(index - 1, 0)]
            delta_nxt = self.delta_list[min(index + 1, len(self) - 1)]
            
            target_exp_diff = torch.matmul(self.exp_base, delta).reshape(-1, 3)
            target_exp_diff_prv = torch.matmul(self.exp_base, delta_prv).reshape(-1, 3)
            target_exp_diff_nxt = torch.matmul(self.exp_base, delta_nxt).reshape(-1, 3)

            return {
                'audio_feature': audio_feature,
                'audio_feature_prv': audio_feature_prv,
                'audio_feature_nxt': audio_feature_nxt,
                'filename': filename,
                'target_exp_diff': target_exp_diff,
                'target_exp_diff_prv': target_exp_diff_prv,
                'target_exp_diff_nxt': target_exp_diff_nxt,
                'index': index,
                'index_prv': max(index - 1, 0),
                'index_nxt': min(index + 1, len(self) - 1)
            }



class Audio2GeometrywithEnergyPitchDataset(Dataset):
    def __init__(self, opt, is_valid=False):
        self.opt = opt
        self.src_dir = opt.src_dir
        self.tgt_dir = opt.tgt_dir
        self.device = opt.device

        self.audio_feature = get_audio_stft(self.src_dir)
        self.audio_energy, self.audio_pitch, _, _, _, _ = get_audio_energy_pitch(self.src_dir)
        _, _, self.e_mean, self.e_std, self.p_mean, self.p_std = get_audio_energy_pitch(self.tgt_dir)
        self.viseme_list = get_viseme_list(self.src_dir)
        
        self.filenames = util.get_file_list(os.path.join(self.tgt_dir, 'alpha'))
        self.start_idx = 0
        self.len = len(self.audio_feature)

        if opt.isTrain:
            self.downsamp_trans = get_downsamp_trans()

            self.delta_list = util.load_coef(os.path.join(self.tgt_dir, 'delta'))
            self.id_base, self.exp_base, self.geo_mean = self.mat_datum()

            self.image_list = util.get_file_list(os.path.join(self.opt.tgt_dir, 'crop'))
            minlen = min(len(self.image_list), len(self.audio_feature))
            self.len = int(minlen * self.opt.train_rate)
            if is_valid:
                self.start_idx = self.len
                self.len = minlen - self.len

        print('dataset size: ', len(self))

    def mat_datum(self):
        mat_data = sio.loadmat(self.opt.matlab_data_path)

        exp_base = torch.from_numpy(mat_data['exp_base']).reshape(-1, 3 * 64)
        exp_base = torch.mm(self.downsamp_trans, exp_base).reshape(-1, 64)

        id_base = torch.from_numpy(mat_data['id_base']).reshape(-1, 3 * 80)
        id_base = torch.mm(self.downsamp_trans, id_base).reshape(-1, 80)

        geo_mean = torch.from_numpy(mat_data['geo_mean']).reshape(-1, 3)
        geo_mean = torch.mm(self.downsamp_trans, geo_mean).reshape(-1, 1)

        return id_base, exp_base, geo_mean

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = index + self.start_idx
        audio_feature = self.audio_feature[index]

        filename = '%05d' % (index + 1) + '.pt'
        audio_energy = (self.audio_energy[index] - self.e_mean) / self.e_std
        audio_pitch = (self.audio_pitch[index] - self.p_mean) / self.p_std
        viseme = self.viseme_list[index]
        
        if not self.opt.isTrain:
            return {'audio_feature': audio_feature, 
                    'viseme': viseme, 
                    'audio_energy': audio_energy, 
                    'audio_pitch': audio_pitch, 
                    'filename': filename}
        else:
            prv_index = max(index - 1, 0)
            nxt_index = min(index + 1, len(self) - 1)
            audio_feature_prv = self.audio_feature[prv_index]
            audio_feature_nxt = self.audio_feature[nxt_index]
            
            delta = self.delta_list[index]
            delta_prv = self.delta_list[prv_index]
            delta_nxt = self.delta_list[nxt_index]
            
            target_exp_diff = torch.matmul(self.exp_base, delta).reshape(-1, 3)
            target_exp_diff_prv = torch.matmul(self.exp_base, delta_prv).reshape(-1, 3)
            target_exp_diff_nxt = torch.matmul(self.exp_base, delta_nxt).reshape(-1, 3)
            
            audio_energy = (self.audio_energy[index] - self.e_mean) / self.e_std
            audio_energy_prv = (self.audio_energy[prv_index] - self.e_mean) / self.e_std
            audio_energy_nxt = (self.audio_energy[nxt_index] - self.e_mean) / self.e_std
            
            audio_pitch = (self.audio_pitch[index] - self.p_mean) / self.p_std
            audio_pitch_prv = (self.audio_pitch[prv_index] - self.p_mean) / self.p_std
            audio_pitch_nxt = (self.audio_pitch[nxt_index] - self.p_mean) / self.p_std

            viseme = self.viseme_list[index]
            viseme_prv = self.viseme_list[prv_index]
            viseme_nxt = self.viseme_list[nxt_index]
            
            return {
                'audio_feature': audio_feature,
                'audio_feature_prv': audio_feature_prv,
                'audio_feature_nxt': audio_feature_nxt,
                'filename': filename,
                'target_exp_diff': target_exp_diff,
                'target_exp_diff_prv': target_exp_diff_prv,
                'target_exp_diff_nxt': target_exp_diff_nxt,
                'audio_energy': audio_energy,
                'audio_energy_prv': audio_energy_prv,
                'audio_energy_nxt': audio_energy_nxt,
                'audio_pitch': audio_pitch,
                'audio_pitch_prv': audio_pitch_prv,
                'audio_pitch_nxt': audio_pitch_nxt,
                'viseme': viseme,
                'viseme_prv': viseme_prv,
                'viseme_nxt': viseme_nxt,
                'index': index,
                'index_prv': max(index - 1, 0),
                'index_nxt': min(index + 1, len(self) - 1)
            }


class ResnetDeltaDataset(Dataset):
    def __init__(self, opt, is_valid=False):
        self.opt = opt
        self.tgt_dir = opt.tgt_dir
        self.device = opt.device
        self.image_list = util.get_file_list(os.path.join(self.tgt_dir, 'crop'))
        
        self.transforms_input = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5141, 0.4074, 0.3588], std=[1.0, 1.0, 1.0])
                                ])

        self.transforms_gt = transforms.ToTensor()
        
        self.downsamp_trans = get_downsamp_trans()
        self.rotation_list = util.load_coef(os.path.join(self.tgt_dir, 'rotation'))
        self.translation_list = util.load_coef(os.path.join(self.tgt_dir, 'translation'))
        self.reference_alpha = torch.load(os.path.join(self.tgt_dir, 'reference_alpha.pt'))
        self.mediapipe_mesh_dict = load_mediapipe_mesh_dict(self.tgt_dir)

        self.id_base, self.exp_base, self.geo_mean = self.mat_datum()

        reference_geometry = self.geo_mean + torch.matmul(self.id_base, self.reference_alpha)
        self.reference_geometry = reference_geometry.reshape(-1, 3)


    def mat_datum(self):
        mat_data = sio.loadmat(self.opt.matlab_data_path)

        exp_base = torch.from_numpy(mat_data['exp_base']).reshape(-1, 3 * 64)
        exp_base = torch.mm(self.downsamp_trans, exp_base).reshape(-1, 64)

        id_base = torch.from_numpy(mat_data['id_base']).reshape(-1, 3 * 80)
        id_base = torch.mm(self.downsamp_trans, id_base).reshape(-1, 80)

        geo_mean = torch.from_numpy(mat_data['geo_mean']).reshape(-1, 3)
        geo_mean = torch.mm(self.downsamp_trans, geo_mean).reshape(-1, 1)

        return id_base, exp_base, geo_mean


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, index):
        image_name = self.image_list[index]
        image = Image.open(image_name).convert('RGB')

        input = self.transforms_input(image)
        mediapipe_mesh_gt = self.mediapipe_mesh_dict[index]
        rotation = self.rotation_list[index]
        translation = self.translation_list[index]
        

        return {'input': input, 
                'reference_geometry': self.reference_geometry, 
                'mediapipe_mesh_gt': mediapipe_mesh_gt,
                'rotation': rotation,
                'translation': translation,
                'image_name': os.path.basename(image_name)}
    
