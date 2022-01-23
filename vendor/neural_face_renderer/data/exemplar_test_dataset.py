import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch

class ExemplarTestDataset(BaseDataset):
    """A dataset class for paired image dataset.

    take label images from 'opt.dataroot/reenact' directory
    take crop lip images from 'opt.dataroot/reenact_crop_lip' directory
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_label = os.path.join(opt.dataroot, 'reenact')  # get the label directory
        self.label_paths = sorted(make_dataset(self.dir_label))  # get label paths
        self.dir_crop_lip = os.path.join(opt.dataroot, 'reenact_crop_lip')
        self.crop_lip_paths = sorted(make_dataset(self.dir_crop_lip))
        
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.Nw = opt.Nw

        # We read the whole dataset into memory during the init period to enhance the IO efficiency
        # First read the whole Images, then Resize and ToTensor
        self.label_Images = []
        self.crop_lip_Images = []

        for i in range(len(self.label_paths)):
            self.label_Images.append(Image.open(self.label_paths[i]).convert('RGB'))
            self.crop_lip_Images.append(Image.open(self.crop_lip_paths[i]).convert('RGB'))

        self.image_size = self.label_Images[0].size


    def get_item_with_index(self, index):
        list_label = []
        transform_params = get_params(self.opt, self.image_size)
        label_transform = get_transform(self.opt, transform_params, grayscale=False)
        crop_lip_transform = label_transform

        if index < self.Nw // 2:
            for i in range(self.Nw // 2 - index):
                label_Image = label_transform(self.label_Images[0])
                crop_lip_Image = crop_lip_transform(self.crop_lip_Images[139])
                list_label.append(label_Image)
                list_label.append(crop_lip_Image)

            for i in range(index + self.Nw // 2 + 1):
                label_Image = label_transform(self.label_Images[i])
                crop_lip_Image = crop_lip_transform(self.crop_lip_Images[139])
                list_label.append(label_Image)
                list_label.append(crop_lip_Image)

        elif index > len(self) - self.Nw // 2 - 1:
            for i in range(index - self.Nw // 2, len(self)):
                label_Image = label_transform(self.label_Images[i])
                crop_lip_Image = crop_lip_transform(self.crop_lip_Images[139])
                list_label.append(label_Image)
                list_label.append(crop_lip_Image)
                
            for i in range(index + self.Nw // 2 - len(self) + 1):
                label_Image = label_transform(self.label_Images[-1])
                crop_lip_Image = crop_lip_transform(self.crop_lip_Images[139])
                list_label.append(label_Image)
                list_label.append(crop_lip_Image)

        else:
            for i in range(index - self.Nw // 2, index + self.Nw // 2 + 1):
                label_Image = label_transform(self.label_Images[i])
                crop_lip_Image = crop_lip_transform(self.crop_lip_Images[139])
                list_label.append(label_Image)
                list_label.append(crop_lip_Image)

        label = torch.cat(list_label, dim=0)
        
        return label
    
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - images which consists of Nw temporal picture
            A_paths (str) - - image path
        """
            
        A = self.get_item_with_index(index)
        A_path = self.label_paths[index]
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.label_paths)