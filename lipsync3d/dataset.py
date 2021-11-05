import sys

from cv2 import normalize
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import os
import torch
import numpy as np
import librosa
from utils import landmarkdict_to_normalized_mesh_tensor, landmarkdict_to_mesh_tensor, get_downsamp_trans
from audiodvp_utils import util
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
from lib.mesh_io import read_obj
from gcn_util.utils import init_sampling
import scipy.io as sio


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
    def __init__(self, opt):
        self.opt = opt
        self.src_dir = opt.src_dir
        self.tgt_dir = opt.tgt_dir
        self.device = opt.device

        stft_path = os.path.join(self.src_dir, 'audio/audio_stft.pt')
        if not os.path.exists(stft_path):
            audio = librosa.load(os.path.join(self.src_dir, 'audio/audio.wav'),16000)[0]
            audio_stft = librosa.stft(audio, n_fft=510, hop_length=160, win_length=480)
            self.audio_stft = torch.from_numpy(np.stack((audio_stft.real, audio_stft.imag)))
            torch.save(self.audio_stft, os.path.join(self.src_dir, 'audio/audio_stft.pt'))
        else:
            self.audio_stft = torch.load(os.path.join(self.src_dir, 'audio/audio_stft.pt'))

        self.filenames = util.get_file_list(os.path.join(self.tgt_dir, 'alpha'))

        if opt.isTrain:
            self.downsamp_trans = get_downsamp_trans()

            self.alpha_list = util.load_coef(os.path.join(self.tgt_dir, 'alpha'))
            self.delta_list = util.load_coef(os.path.join(self.tgt_dir, 'delta'))
            self.reference_alpha = torch.load(os.path.join(self.tgt_dir, 'reference_alpha.pt'))
            
            self.id_base, self.exp_base = self.mat_datum()

            self.image_list = util.get_file_list(os.path.join(self.opt.tgt_dir, 'crop'))
            minlen = min(len(self.image_list), self.audio_stft.shape[2] // 4)
            train_idx = int(minlen * self.opt.train_rate)
            self.filenames = self.filenames[:train_idx]

        print('training set size: ', len(self))

    def mat_datum(self):
        mat_data = sio.loadmat(self.opt.matlab_data_path)
        id_base = torch.from_numpy(mat_data['id_base'])
        exp_base = torch.from_numpy(mat_data['exp_base'])
        
        return id_base, exp_base

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
            return {'audio_feature': audio_feature, 'filename': filename}

        else:
            alpha = self.alpha_list[index]
            delta = self.delta_list[index]

            geometry = torch.matmul(self.id_base, alpha - self.reference_alpha) + torch.matmul(self.exp_base, delta)
            geometry = geometry.reshape(35709, 3)

            geometry = torch.mm(self.downsamp_trans, geometry)

            return {
                'audio_feature': audio_feature, 
                'filename': filename,
                'geometry': geometry
            }
