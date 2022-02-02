import sys
sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

from audiodvp_utils import util, audio
import numpy as np
import math
import torch
import os
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
from tqdm import tqdm
import cv2
import pickle
from lipsync3d import lpc
import librosa
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lipsync3d.audio.stft import TacotronSTFT
import pyworld as pw
import tgt

# Input :
#       reference(dictionary from vertex idx to normalized landmark, dict[idx] = [x, y, z]) : landmark of reference frame.
#       target(dictionary from vertex idx to normalized landmark, dict[idx] = [x, y, z]) : landmark of target frame.
# Output : 
#       R : 3x3 Rotation matrix(np.array)
#       c : scale value(float)
#       t : 3x1 translation matrix(np.array)

def Umeyama_algorithm(reference, target):
    # idx 2 -> nose, 130 -> left eye, 359 -> right eye
    idx_list = [2, 94, 19, 1, 4, 5, 195, 197, 6, 168, 8, 9, 151, 10, 109, 108, 67, 69, 103, 104, 54, 68, 338, 337, 297, 299, 332, 333, 284, 298, 130, 243, 244, 359, 362, 463,
                21, 71, 162, 139, 156, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 301, 251, 55, 285, 193, 417, 122, 351, 196, 419, 3, 248, 51, 281,
                45, 275, 44, 274, 220, 440, 134, 363, 236, 456]
    # idx_list = [19, 243, 463]
    ref_points = []
    tgt_points = []

    for idx in idx_list:
        ref_points.append(reference[idx])
        tgt_points.append(target[idx])

    ref_points = np.array(ref_points)
    tgt_points = np.array(tgt_points)

    ref_mu = ref_points.mean(axis=0)
    tgt_mu = tgt_points.mean(axis=0)
    ref_var = ref_points.var(axis=0).sum()
    tgt_var = tgt_points.var(axis=0).sum()
    n, m = ref_points.shape
    covar = np.matmul((ref_points - ref_mu).T, tgt_points - tgt_mu) / n
    det_covar = np.linalg.det(covar)
    u, d, vh = np.linalg.svd(covar)
    detuv = np.linalg.det(u) * np.linalg.det(vh.T)
    cov_rank = np.linalg.matrix_rank(covar)
    S = np.identity(m)

    if cov_rank > m - 1:
        if det_covar < 0:
            S[m - 1, m - 1] = -1
    else: 
        if detuv < 0:
            S[m - 1, m - 1] = -1

    R = np.matmul(np.matmul(u, S), vh)
    c = (1 / tgt_var) * np.trace(np.matmul(np.diag(d), S))
    t = ref_mu.reshape(3, 1) - c * np.matmul(R, tgt_mu.reshape(3, 1))

    return R, t, c


def landmark_to_dict(landmark_list):
    landmark_dict = {}
    for idx, landmark in enumerate(landmark_list):
        landmark_dict[idx] = [landmark.x, landmark.y, landmark.z]

    return landmark_dict

def landmarkdict_to_normalized_mesh_tensor(landmark_dict):
    vertex_list = []
    for idx, coord in landmark_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        vertex_list.append(coord)
    
    if not ('R' in landmark_dict):
        return torch.tensor(vertex_list)
    
    R = torch.from_numpy(landmark_dict['R']).float()
    t = torch.from_numpy(landmark_dict['t']).float()
    c = float(landmark_dict['c'])
    vertices = torch.tensor(vertex_list).transpose(0, 1)
    norm_vertices = (c * torch.matmul(R, vertices) + t).transpose(0, 1)
    return norm_vertices


def landmarkdict_to_mesh_tensor(landmark_dict):
    vertex_list = []
    face_oval = get_face_oval_indices()
    for idx, coord in landmark_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        if idx in face_oval:
            continue
        vertex_list.append(coord)

    vertices = torch.tensor(vertex_list)
    return vertices

def mesh_tensor_to_landmarkdict(mesh_tensor):
    landmark_dict = {}
    for i in range(mesh_tensor.shape[0]):
        landmark_dict[i] = mesh_tensor[i].tolist()
    
    return landmark_dict


def draw_mesh_image(mesh_dict, save_path, image_rows, image_cols):
    connections = mp_face_mesh.FACEMESH_TESSELATION
    drawing_spec = mp_drawing.DrawingSpec(color= mp_drawing.BLACK_COLOR, thickness=1, circle_radius=1)

    idx_to_coordinates = {}
    for idx, coord in mesh_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        x_px = min(math.floor(coord[0]), image_cols - 1)
        y_px = min(math.floor(coord[1]), image_rows - 1)
        landmark_px = (x_px, y_px)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    
    white_image = np.zeros([image_rows, image_cols, 3], dtype=np.uint8)
    white_image[:] = 255
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]

        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            cv2.line(white_image, 
                idx_to_coordinates[start_idx],
                idx_to_coordinates[end_idx], 
                drawing_spec.color,
                drawing_spec.thickness
            )
    cv2.imwrite(save_path, white_image)


def draw_mesh_images(mesh_dir, save_dir, image_rows, image_cols):
    mesh_filename_list = util.get_file_list(mesh_dir)

    for mesh_filename in tqdm(mesh_filename_list):
        mesh_dict = torch.load(mesh_filename)
        save_path = os.path.join(save_dir, os.path.basename(mesh_filename)[:-3] + '.png')
        draw_mesh_image(mesh_dict, save_path, image_rows, image_cols)
    
    return

def get_face_oval_indices():
    face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]  # 36
    return face_oval


def downsamp_mediapipe_mesh(mediapipe_mesh):
    face_oval = get_face_oval_indices()
    
    mediapipe_mesh = mediapipe_mesh.tolist()
    downsamp_mesh = []
    for i in range(len(mediapipe_mesh)):
        if i in face_oval:
            continue
        downsamp_mesh.append(mediapipe_mesh[i])
    
    return torch.tensor(downsamp_mesh)

def get_downsamp_trans():
    # output: 442 x 35709
    
    old_indices = [8095, 8193, 8202, 8542, 8191, 8186, 8169, 13301, 8143, 8129, 29880, 8098, 8102, 8105, 8108, 8110, 8113, 8115, 8241, 8196, 
                8921, 24308, 11371, 12016, 12532, 13691, 10854, 11603, 10960, 12377, 13153, 14599, 34011, 13815, 23572, 16141, 11917, 9175, 
                8823, 10142, 10655, 9544, 10151, 10933, 8434, 8431, 14178, 10354, 11016, 11013, 13852, 8427, 11593, 12756, 31364, 9333, 10448, 
                11824, 27580, 9757, 9521, 10667, 10410, 31060, 11148, 30459, 30455, 30442, 31229, 30421, 14818, 23014, 9058, 9780, 10274, 9759, 
                10538, 10160, 10283, 9397, 9548, 9068, 8825, 8841, 8715, 8713, 8710, 8708, 9548, 9670, 9672, 9918, 11423, 24127, 8199, 9911, 
                10035, 9283, 10894, 9402, 11259, 12296, 11145, 30841, 30783, 30726, 10295, 30208, 30174, 30168, 13177, 15379, 10337, 14965, 
                9734, 9632, 22035, 14480, 13450, 12158, 11251, 10475, 8649, 15917, 15867, 8437, 10488, 23833, 9849, 11275, 14460, 10115, 27248, 
                10208, 9027, 34776, 34690, 23863, 35438, 23424, 34015, 8439, 11267, 22663, 12526, 12010, 10287, 16062, 33911, 34154, 34363, 29912, 
                33821, 11495, 10980, 10463, 22645, 10842, 11355, 11870, 12516, 13164, 24328, 12914, 8207, 10773, 9636, 9408, 8158, 34373, 34166, 
                33922, 35110, 10457, 9137, 33831, 34003, 27360, 9069, 9190, 9193, 9195, 9560, 10404, 10533, 10662, 11816, 14643, 9131, 9468, 9950, 
                9911, 15055, 8754, 9685, 8182, 8656, 8176, 9987, 33828, 8246, 8846, 11452, 11538, 10428, 12828, 12061, 13485, 33903, 10750, 11847, 
                34159, 12472, 16207, 13511, 23902, 12457, 9740, 9394, 10760, 8911, 9944, 10826, 11599, 12503, 13537, 15364, 23588, 13957, 13055, 
                12151, 11246, 10599, 10090, 27207, 10891, 9023, 8915, 9041, 9280, 10635, 8801, 8682, 9960, 9600, 9363, 13554, 14190, 7701, 2609, 
                7601, 19151, 4808, 4292, 3519, 2226, 5452, 4523, 5170, 3621, 2848, 1587, 33626, 1962, 20090, 34, 4452, 7135, 7382, 6152, 5511, 6658, 
                6030, 5402, 7600, 7831, 1552, 5726, 5357, 5354, 2518, 7827, 4386, 3097, 28373, 6809, 5690, 4487, 17850, 6510, 6880, 5261, 5520, 28649,
                5360, 29359, 29355, 29313, 28532, 29378, 27953, 20824, 7258, 6409, 5770, 6512, 5390, 6039, 5649, 6997, 6290, 7148, 7628, 7641, 7635, 
                7633, 7630, 7628, 6786, 6664, 6423, 6303, 4989, 18064, 5904, 5906, 7123, 5493, 7002, 4955, 3928, 5228, 28910, 29016, 29054, 6182, 
                29604, 29624, 29618, 2874, 819, 5965, 1177, 6363, 6749, 21778, 1856, 2889, 4176, 5076, 5717, 7569, 68, 18, 7957, 5732, 19319, 6356, 
                5100, 1316, 6253, 18344, 6089, 7227, 32763, 32905, 17026, 32074, 20329, 33631, 7960, 5092, 21245, 3641, 4157, 5912, 21936, 33732, 
                33486, 33279, 4674, 5190, 5704, 21099, 5311, 4792, 4146, 3501, 2858, 19041, 3126, 5630, 6631, 7128, 33236, 33454, 33742, 32507, 
                5699, 6976, 33637, 17176, 7148, 7030, 7032, 7035, 6920, 5775, 5516, 5257, 4608, 1631, 6970, 6826, 6336, 5904, 1140, 7430, 6682, 
                7696, 6374, 7526, 4890, 4718, 5800, 3557, 4338, 2795, 33738, 5607, 4382, 33468, 3975, 21824, 2821, 18742, 3831, 6369, 6994, 5746, 
                7351, 6203, 5295, 4391, 3488, 2330, 804, 19848, 2234, 3139, 4170, 4941, 5584, 6098, 18692, 5490, 7102, 7356, 7481, 7120, 5750, 7721, 
                7841, 6467, 6593, 6839, 2215, 1950, 11747, 10973, 11480, 12391, 11755, 4409, 3505, 4529, 5054, 4288]
    
    face_oval = get_face_oval_indices()

    new_indices = []
    for i in range(len(old_indices)):
        if i in face_oval:
            continue
        new_indices.append(old_indices[i])
        
    downsamp_trans = torch.zeros((len(new_indices), 35709)) # 478 x 35709

    for i in range(len(new_indices)):
        downsamp_trans[i, new_indices[i]] = 1
    
    return downsamp_trans

    
def normalized_to_pixel_coordinates(landmark_dict, image_width, image_height):
    def is_valid_normalized_value(value):
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    landmark_pixel_coord_dict = {}

    for idx, coord in landmark_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue

        if not (is_valid_normalized_value(coord[0]) and
                is_valid_normalized_value(coord[1])):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = coord[0] * image_width
        y_px = coord[1] * image_height
        z_px = coord[2] * image_width
        landmark_pixel_coord_dict[idx] = [x_px, y_px, z_px]
    return landmark_pixel_coord_dict


def load_mediapipe_mesh_dict(target_dir):
    mediapipe_mesh_dict_path = os.path.join(target_dir, 'mediapipe_mesh.pkl')

    if not os.path.exists(mediapipe_mesh_dict_path):
        image_list = util.get_file_list(os.path.join(target_dir, 'crop'))
        
        mediapipe_mesh_dict = {}
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

            for i in tqdm(range(len(image_list))):
                image = cv2.imread(image_list[i])
                annotated_image = image.copy()
                image_rows, image_cols, _ = image.shape
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                target_dict = landmark_to_dict(results.multi_face_landmarks[0].landmark)
                target_dict = normalized_to_pixel_coordinates(target_dict, image_cols, image_rows)
                target_dict = landmarkdict_to_mesh_tensor(target_dict)
                mediapipe_mesh_dict[i] = target_dict[:, :2] # 442 x 2
    
        with open(mediapipe_mesh_dict_path, 'wb') as f:
            pickle.dump(mediapipe_mesh_dict, f)

    with open(mediapipe_mesh_dict_path, 'rb') as f:
        mediapipe_mesh_dict = pickle.load(f)

    return mediapipe_mesh_dict


def get_train_data_statistic(opt):
    delta_list = util.load_coef(os.path.join(opt.tgt_dir, 'delta'))
    downsamp_trans = get_downsamp_trans()

    mat_data = sio.loadmat(opt.matlab_data_path)
    exp_base = torch.from_numpy(mat_data['exp_base']).reshape(-1, 3 * 64)
    exp_base = torch.mm(downsamp_trans, exp_base).reshape(-1, 64)

    exp_list = []
    for delta in delta_list:
        target_exp_diff = torch.matmul(exp_base, delta).reshape(1, -1)
        exp_list.append(target_exp_diff)
    
    exp_tensor = torch.cat(exp_list, dim=0)
    exp_numpy = exp_tensor.numpy()
    SS= StandardScaler()
    scaled_exp = SS.fit_transform(exp_numpy)
    pca = PCA(n_components=150)
    pca.fit(scaled_exp)
    
    mean = torch.from_numpy(SS.mean_).to(opt.device).float()
    scale = torch.from_numpy(SS.scale_).to(opt.device).float()
    components = torch.from_numpy(pca.components_).transpose(0, 1).to(opt.device).float()
    
    return mean, scale, components
    

def get_audio_stft(src_dir):
    save_path = os.path.join(src_dir, 'audio/audio_stft.pt')
    audio_path = os.path.join(src_dir, 'audio/audio.wav')

    if os.path.exists(save_path):
        return torch.load(save_path)
    
    audio = librosa.load(audio_path,16000)[0]
    audio_stft = librosa.stft(audio, n_fft=510, hop_length=160, win_length=480)
    audio_stft_real_imag = torch.from_numpy(np.stack((audio_stft.real, audio_stft.imag)))
    
    audio_stft = []
    print('audio pre-processing...')
    for idx in tqdm(range(audio_stft_real_imag.shape[2] // 4)):
        audio_idx = idx * 4
        audio_feature_list = []
        for i in range(audio_idx - 12, audio_idx + 12):
            if i < 0:
                audio_feature_list.append(audio_stft_real_imag[:, :, 0])
            elif i >= audio_stft_real_imag.shape[2]:
                audio_feature_list.append(audio_stft_real_imag[:, :, -1])
            else:
                audio_feature_list.append(audio_stft_real_imag[:, :, i])
                
        audio_feature = torch.stack(audio_feature_list, 2)
        audio_stft.append(audio_feature)
    
    torch.save(audio_stft, os.path.join(src_dir, 'audio/audio_stft.pt'))
    
    return audio_stft


def get_mel_spectogram(src_dir):
    save_path = os.path.join(src_dir, 'audio/audio_mel.pt')
    audio_path = os.path.join(src_dir, 'audio/audio.wav')

    if os.path.exists(save_path):
        return torch.load(save_path)
    
    wav, sr = librosa.load(audio_path,16000)
    wav = torch.from_numpy(wav).unsqueeze(0)
    taco = TacotronSTFT(510, 160, 480, 80, sr, 0, 8000)
    mel, energy = taco.mel_spectrogram(wav)
    mel = mel.squeeze()
    
    audio_mel = []
    print('audio pre-processing...')
    for idx in tqdm(range(mel.shape[-1] // 4)):
        audio_idx = idx * 4
        audio_feature_list = []
        for i in range(audio_idx - 12, audio_idx + 12):
            if i < 0:
                audio_feature_list.append(mel[:, 0])
            elif i >= mel.shape[-1]:
                audio_feature_list.append(mel[:, -1])
            else:
                audio_feature_list.append(mel[:, i])
                
        audio_feature = torch.stack(audio_feature_list, 1).float()
        audio_mel.append(audio_feature.unsqueeze(0))
    
    torch.save(audio_mel, save_path)
    
    return audio_mel


def get_autocorrelation_coefficients(src_dir):
    save_path = os.path.join(src_dir, 'audio/audio_autocorrelation.pt')
    audio_path = os.path.join(src_dir, 'audio/audio.wav')
    
    if os.path.exists(save_path):
        return torch.load(save_path)
        
    waveform, sampleRate = librosa.load(audio_path, 16000)
    waveform = torch.from_numpy(waveform).view(1, -1)
        
    audioFrameLen = int(.008 * 16000 * (64 + 1))
    LPC = lpc.LPCCoefficients(
        sampleRate,
        .016,
        .5,
        order=31  # 32 - 1
    )

    autocorrelation_coefficients = []
    
    count = int(25 * waveform.shape[1] / sampleRate)
    for idx in tqdm(range(count)):
        start_idx = 640 * idx - int(audioFrameLen / 2)
        end_idx = 640 * idx + int(audioFrameLen / 2)
        indices = []
        if start_idx < 0:
            for _ in range(start_idx, 0):
                indices.append(0)
            indices = indices + list(range(0, end_idx))
            autocorrelation_coefficients.append(LPC(waveform[:, indices]).transpose(1, 2))
        elif (start_idx > 0) and end_idx < waveform.shape[1]:
            autocorrelation_coefficients.append(LPC(waveform[:, start_idx:end_idx]).transpose(1, 2))
        else:
            indices = list(range(start_idx, waveform.shape[1]))
            for _ in range(waveform.shape[1], end_idx):
                indices.append(waveform.shape[1] - 1)
            autocorrelation_coefficients.append(LPC(waveform[:, indices]).transpose(1, 2))
            
    torch.save(autocorrelation_coefficients, save_path)
    return autocorrelation_coefficients


def get_audio_energy_pitch(src_dir):
    save_path = os.path.join(src_dir, 'audio/audio_energy_pitch.pt')
    audio_path = os.path.join(src_dir, 'audio/audio.wav')

    if os.path.exists(save_path):
        return torch.load(save_path)
    
    wav, sr = librosa.load(audio_path,16000)
    torch_wav = torch.from_numpy(wav).unsqueeze(0)
    taco = TacotronSTFT(510, 160, 480, 80, sr, 0, 8000)
    mel, energy = taco.mel_spectrogram(torch_wav)
    
    pitch, t = pw.dio(
            wav.astype(np.float64),
            sr,
            frame_period=160 / sr * 1000,
        )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sr)
    pitch = torch.tensor(pitch).float().unsqueeze(0)
    audio_energy = []
    audio_pitch = []
    print('audio energy pitch pre-processing...')
    for idx in tqdm(range(energy.shape[-1] // 4)):
        audio_idx = idx * 4
        audio_energy_list = []
        audio_pitch_list = []
        
        for i in range(audio_idx - 12, audio_idx + 12):
            if i < 0:
                audio_energy_list.append(energy[:, 0])
                audio_pitch_list.append(pitch[:, 0])
            elif i >= energy.shape[-1]:
                audio_energy_list.append(energy[:, -1])
                audio_pitch_list.append(pitch[:, -1])
            else:
                audio_energy_list.append(energy[:, i])
                audio_pitch_list.append(pitch[:, i])
                
        audio_e = torch.stack(audio_energy_list, 1)
        audio_p = torch.stack(audio_pitch_list, 1)
        audio_energy.append(audio_e)
        audio_pitch.append(audio_p)
    
    e_mean = energy.mean(dim=1)
    e_std = energy.std(dim=1)
    p_mean = pitch.mean(dim=1)
    p_std = pitch.std(dim=1)
    
    torch.save((audio_energy, audio_pitch, e_mean, e_std, p_mean, p_std), save_path)
    
    return audio_energy, audio_pitch, e_mean, e_std, p_mean, p_std


def get_viseme_list(src_dir):
    save_path = os.path.join(src_dir, 'audio/audio_viseme.pt')
    audio_path = os.path.join(src_dir, 'audio/audio.wav')

    if os.path.exists(save_path):
        return torch.load(save_path)
    
    viseme_mapping = {'spn': 0, 'AA0': 1, 'AA1': 1, 'AA2': 1, \
                    'AH0': 2, 'AH1': 2, 'AH2': 2, 'HH': 2, \
                    'AO0': 3, 'AO1': 3, 'AO2': 3, \
                    'AW0': 4, 'AW1': 4, 'AW2': 4, 'OW0': 4, 'OW1': 4, 'OW2': 4, \
                    'OY0': 5, 'OY1': 5, 'OY2': 5, 'UH0': 5, 'UH1': 5, 'UH2': 5, 'UW0': 5, 'UW1': 5, 'UW2': 5, \
                    'EH0': 6, 'EH1': 6, 'EH2': 6, 'AE0': 6, 'AE1': 6, 'AE2': 6, \
                    'IH0': 7, 'IH1': 7, 'IH2': 7, 'AY0': 7, 'AY1': 7, 'AY2': 7, \
                    'EY0': 8, 'EY1': 8, 'EY2': 8, \
                    'Y': 9, 'IY0': 9, 'IY1': 9, 'IY2': 9, \
                    'R': 10, 'ER0': 10, 'ER1': 10, 'ER2': 10, \
                    'L': 11, \
                    'W': 12, \
                    'M': 13, 'P': 13, 'B': 13, \
                    'N': 14, 'NG': 14, 'DH': 14, 'D': 14, 'G': 14, 'T': 14, 'Z': 14, 'ZH': 14, 'TH': 14, 'K': 14, 'S': 14, \
                    'CH': 15, 'JH': 15, 'SH': 15, \
                    'F': 16, 'V': 16, \
                    '': 17}
    
    wav = librosa.load(audio_path,16000)[0]
    audio_stft = librosa.stft(wav, n_fft=510, hop_length=160, win_length=480)
    file_path = os.path.join(src_dir, 'flist.txt')
    
    phoneme_list = [None for _ in range(audio_stft.shape[-1])]
    
    total_time = 0

    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            fname = line.split(' ')[-1][1:-5]
            tg_name = "42_{}.TextGrid".format(fname)
            tg_path = os.path.join(src_dir, "tcdtimit4_textgrid/{}".format(tg_name))   
            textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
            tier = textgrid.get_tier_by_name('phones')
            for t in tier._objects:
                s, e, p = t.start_time, t.end_time, t.text
                if int((total_time + e - s) * 100) < len(phoneme_list):
                    for i in range(int(total_time * 100), int((total_time + e - s) * 100)+1):
                        phoneme_list[i] = viseme_mapping[p]
                else:
                    for i in range(int(total_time * 100), len(phoneme_list)):
                        phoneme_list[i] = viseme_mapping[p]
                    
                total_time += e - s
    
    
    audio_viseme = []
    print('audio phoneme-viseme pre-processing...')
    for idx in tqdm(range(len(phoneme_list) // 4)):
        audio_idx = idx * 4
        audio_viseme_list = []
        
        for i in range(audio_idx - 12, audio_idx + 12):
            if i < 0:
                audio_viseme_list.append(phoneme_list[0])
            elif i >= len(phoneme_list):
                audio_viseme_list.append(phoneme_list[-1])
            else:
                audio_viseme_list.append(phoneme_list[i])
        
        one_hot = torch.zeros((18, 24))
        for i in range(len(audio_viseme_list)):
            one_hot[audio_viseme_list[i], i]= 1
        
        audio_viseme.append(one_hot)
    torch.save((audio_viseme), save_path)
    
    return audio_viseme


