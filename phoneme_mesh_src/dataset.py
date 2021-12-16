import sys

sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import os
import torch
import numpy as np
import librosa
from utils import landmarkdict_to_normalized_mesh_tensor, landmarkdict_to_mesh_tensor
from audiodvp_utils import util
from torch.utils.data import Dataset
from natsort import natsorted
import torchvision.transforms as transforms
import cv2
from PIL import Image

class Lipsync3DDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.src_dir = opt.src_dir
        self.tgt_dir = opt.tgt_dir
        self.y_size = 150

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

        self.tts_data = torch.load(os.path.join(self.src_dir, 'tts_features.dat'))
        for k in self.tts_data:
            if k == 'mask' or k == 'src_seq':
                self.tts_data[k] = torch.from_numpy(self.tts_data[k])
            else:
                self.tts_data[k] = torch.from_numpy(self.tts_data[k]).float()
        print("LOADED TTS DATA !")
        
        if opt.use_texture:
            # -------------------------------------------- Added by Jonghoon Shin
            self.transform = transforms.Compose([
                transforms.ToTensor()
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            self.texture_image = [os.path.join(self.tgt_dir, 'texture_images', x) for x in natsorted(os.listdir(os.path.join(self.tgt_dir, 'texture_images')))]
            self.texture_mesh = [os.path.join(self.tgt_dir, 'texture_mesh', x) for x in natsorted(os.listdir(os.path.join(self.tgt_dir, 'texture_mesh')))]
            self.mouthRegion = [62, 78, 191, 95, 80, 88, 81, 178, 82, 87, 13, 14, 312, 317, 311, 402, 310, 318, 415, 324, 308, 293]
            # reference_texture_mesh = torch.load(os.path.join(self.tgt_dir, 'texture_reference_mesh.pt'))
            reference_texture_image = np.array(Image.open(os.path.join(self.tgt_dir, 'reference_texture.jpg')).convert('RGB'))
            # x = np.average(reference_texture_mesh[self.mouthRegion, 0]).astype(int)
            # y = np.average(reference_texture_mesh[self.mouthRegion, 1]).astype(int)
            self.reference_mouth_image = self.transform(reference_texture_image[self.y_size:, :, :])
            # -------------------------------------------- Added by Jonghoon Shin
        
        if opt.isTrain:
            minlen = min(len(self.mesh_dict_list), self.audio_stft.shape[2] // 4)
            train_idx = int(minlen * self.opt.train_rate)
            self.mesh_dict_list = self.mesh_dict_list[:train_idx]
            self.filenames = self.filenames[:train_idx]
            if opt.use_texture:
                # -------------------------------------------- Added by Jonghoon Shin
                self.texture_image = self.texture_image[:train_idx]
                self.texture_mesh = self.texture_mesh[:train_idx]
                # -------------------------------------------- Added by Jonghoon Shin

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


        def get_tts_feature(key):
            feature_list = []
            if key == "mel_target" or key == "mel_output":
                def idx_fn(full_feature, safe_idx):
                    feature_fr = full_feature[safe_idx, :]
                    if self.tts_data["mask"][safe_idx] < 0.5:
                        feature_fr = feature_fr - 5.0 # low dB for silent frame
                    return feature_fr
                stack_idx = 1
            elif key == "src_seq":
                def idx_fn(full_feature, safe_idx):
                    feature_fr = full_feature[safe_idx]
                    if self.tts_data["mask"][safe_idx] < 0.5:
                        feature_fr = torch.zeros_like(feature_fr)
                    else:
                        feature_fr += 1 # set 0 as mask id
                    return feature_fr
                stack_idx = 0
            elif key == "enc_pre":
                def idx_fn(full_feature, safe_idx):
                    feature_fr = full_feature[safe_idx, :]
                    return feature_fr
                stack_idx = 1
            else:
                raise RuntimeError # feature not supported
            for i in range(audio_idx - 12, audio_idx + 12):
                if i < 0:
                    safe_idx = 0
                elif i >= len(self.tts_data["src_seq"]):
                    safe_idx = -1
                else:
                    safe_idx = i
                feature_fr = idx_fn(self.tts_data[key], safe_idx)
                feature_list.append(feature_fr)
            feature = torch.stack(feature_list, stack_idx)
            return feature

        self.tts_feature_req = [] 
        if self.opt.use_mel_target:
            self.tts_feature_req.append('mel_target')
        if self.opt.use_src_seq:
            self.tts_feature_req.append('src_seq')
        if self.opt.use_enc_pre:
            self.tts_feature_req.append('enc_pre')

        tts_feature_dict = {}
        for k in self.tts_feature_req:
            feature = get_tts_feature(k)
            tts_feature_dict[k] = feature

        filename = os.path.basename(self.filenames[index])

        # make data item (refactored by Eunhyuk)
        data_dict = {}
        landmark_dict = self.mesh_dict_list[index]
        normalized_mesh = landmarkdict_to_normalized_mesh_tensor(landmark_dict)
        data_dict["normalized_mesh"] = normalized_mesh
        data_dict["audio_feature"] = audio_feature
        data_dict["filename"] = filename
        data_dict["reference_mesh"] = self.reference_mesh
        R = torch.from_numpy(landmark_dict['R']).float()
        t = torch.from_numpy(landmark_dict['t']).float()
        c = float(landmark_dict['c'])
        data_dict["R"] = R
        data_dict["t"] = t
        data_dict["c"] = c

        if not self.opt.isTrain:
            if self.opt.use_texture:
                data_dict["reference_mouth"] = self.reference_mouth_image
        else:
            if self.opt.use_texture:
                texture = np.array(Image.open(self.texture_image[index]).convert('RGB'))
                texture_mouth = self.transform(texture[self.y_size:, :, :])
                if index == 0:
                    previous_texture = torch.zeros(3, 150, 300)
                else:
                    previous_texture = cv2.imread(self.texture_image[index - 1])
                    previous_texture = self.transform(previous_texture[self.y_size:, :, :])
                data_dict["reference_mouth"] = self.reference_mouth_image
                data_dict["texture_mouth"] = texture_mouth
                data_dict["previous_texture"] = previous_texture

        for k in self.tts_feature_req:
            data_dict[k] = tts_feature_dict[k]

        return data_dict
                

