import os
import pickle
from PIL import Image
import torch
from torchvision import transforms

from audiodvp_utils import util
from datasets.base_dataset import BaseDataset


class SingleDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.image_list = util.get_file_list(os.path.join(self.opt.data_dir, 'crop'))

        self.transforms_input = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5141, 0.4074, 0.3588], std=[1.0, 1.0, 1.0])
                                ])

        self.landmark_dict = self.load_landmark_dict()
        # self.face_emb_dict = self.load_face_emb_dict()

        self.transforms_gt = transforms.ToTensor()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image = Image.open(image_name).convert('RGB')

        input = self.transforms_input(image)
        gt = self.transforms_gt(image)
        landmark_gt = torch.tensor(self.landmark_dict[image_name])
        # face_emb = self.face_emb_dict[image_name]

        return {'input': input, 'gt': gt, 'landmark_gt': landmark_gt, 'image_name': os.path.basename(image_name)}

    def load_landmark_dict(self):
        landmark_path = os.path.join(self.opt.data_dir, 'landmark.pkl')

        if not os.path.exists(landmark_path):
            util.landmark_detection(self.image_list, landmark_path)

        with open(landmark_path, 'rb') as f:
            landmark_dict = pickle.load(f)

        return landmark_dict

    def load_face_emb_dict(self):
        face_emb_dir = os.path.join(self.opt.data_dir, 'face_emb.pkl')

        if not os.path.exists(face_emb_dir):
            util.extract_face_emb(self.image_list, face_emb_dir, self.transforms_input)
        
        with open(face_emb_dir, 'rb') as f:
            face_emb_dict = pickle.load(f)
        
        return face_emb_dict
