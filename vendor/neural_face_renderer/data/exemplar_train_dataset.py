import os.path
from re import L
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch

class ExemplarTrainDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, 'nfr/AB/train')  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB))  # get image paths
        self.dir_crop_lip = os.path.join(opt.dataroot, 'crop_lip')
        self.crop_lip_paths = sorted(make_dataset(self.dir_crop_lip))
        
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.Nw = opt.Nw

        # We read the whole dataset into memory during the init period to enhance the IO efficiency
        # First read the whole Images, then Resize and ToTensor
        self.AB_Images = []
        self.A_Images = []
        self.B_Images = []
        self.crop_lip_Images = []
        image = Image.open(self.AB_paths[0]).convert('RGB')
        _, h = image.size
        self.image_size = (h, h)
        

        for i in range(len(self.AB_paths)):
            self.AB_Images.append(Image.open(self.AB_paths[i]).convert('RGB'))
            current_image = self.AB_Images[i]
            w, h = current_image.size
            w2 = int(w / 2)
            self.A_Images.append(self.AB_Images[i].crop((0, 0, w2, h)))
            self.B_Images.append(self.AB_Images[i].crop((w2, 0, w, h)))
            self.crop_lip_Images.append(Image.open(self.crop_lip_paths[i]).convert('RGB'))


    def get_item_with_index(self, index):
        transform_params = get_params(self.opt, self.image_size)
        A_transform = get_transform(self.opt, transform_params, grayscale=False)
        B_transform = get_transform(self.opt, transform_params, grayscale=False)
        crop_lip_transform = A_transform
        list_B = []

        if index < self.Nw // 2:
            for i in range(self.Nw // 2 - index):
                B_Image = B_transform(self.B_Images[0])
                crop_lip_Image = crop_lip_transform(self.crop_lip_Images[0])
                list_B.append(B_Image)
                list_B.append(crop_lip_Image)

            for i in range(index + self.Nw // 2 + 1):
                B_Image = B_transform(self.B_Images[i])
                crop_lip_Image = crop_lip_transform(self.crop_lip_Images[i])
                list_B.append(B_Image)
                list_B.append(crop_lip_Image)

        elif index > len(self) - self.Nw // 2 - 1:
            for i in range(index - self.Nw // 2, len(self)):
                B_Image = B_transform(self.B_Images[i])
                crop_lip_Image = crop_lip_transform(self.crop_lip_Images[i])
                list_B.append(B_Image)
                list_B.append(crop_lip_Image)

            for i in range(index + self.Nw // 2 - len(self) + 1):
                B_Image = B_transform(self.B_Images[-1])
                crop_lip_Image = crop_lip_transform(self.crop_lip_Images[-1])
                list_B.append(B_Image)
                list_B.append(crop_lip_Image)

        else:
            for i in range(index - self.Nw // 2, index + self.Nw // 2 + 1):
                B_Image = B_transform(self.B_Images[i])
                crop_lip_Image = crop_lip_transform(self.crop_lip_Images[i])
                list_B.append(B_Image)
                list_B.append(crop_lip_Image)

        A = A_transform(self.A_Images[index])
        B = torch.cat(list_B, dim=0)
        AB_path = self.AB_paths[index]

        return A, B, AB_path
    
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - ground truth image
            B (tensor) - - images which consists of Nw temporal picture
            A_paths (str) - - image paths
            B_paths (str) - - image paths
        """
            
        A, B, path = self.get_item_with_index(index)

        return {'A': A, 'B': B, 'A_paths': path, 'B_paths': path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
