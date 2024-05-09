import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import cv2
import numpy as np

class UnalignedOneHotsegOnecycleDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        # SHUNXING added to transform grayscale image only
        self.transform_grayscale = get_transform(self.opt, grayscale=True) 
        # SHUNXING added to transform applies nearest neighbor interpolation to set curImgIsLabel
        self.transform_label = get_transform(self.opt, grayscale=False, curImgIsLabel=True)
        
        self.patch_size = self.opt.patch_size

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        #SHUNXING
        # comment A_img RGB because the input has 4 channels: RGB + seg mask
        A_img = Image.open(A_path)#.convert('RGB') 
        B_img = Image.open(B_path).convert('RGB')
        
        # apply image transformation
        #A = self.transform_A(A_img)
        seed_A = np.random.randint(2147483647)

        transform_B_color_jitter = get_transform(self.opt, grayscale=False, curImgIsHE = True) # apply colorjitter data augmentation on vHE, could turn
        # self.reset_random_seed(seed_B)
        B = transform_B_color_jitter(B_img)
       
        tensor_A_list = []
        #contain_list = [2,17,12] # for testing MxIF dataset 
        contain_list = [0,1,2] # 0 is red channel of realHE, 1 is green channel of realHE, 2 is blue channel
        for i in range(0, len(contain_list)):
            # select only 11 markers
            roi_idx = contain_list[i]
            
            A_tmp = A_img.crop((roi_idx * self.patch_size, 0, roi_idx * self.patch_size + self.patch_size, self.patch_size))
            self.reset_random_seed(seed_A)  # make sure the random seed is reset so that random generator from torch gets same number
            A_tmp = self.transform_grayscale(A_tmp)
            tensor_A_list.append(A_tmp)

        # for creating label
        # make sure the random seed is reset so that random generator from torch gets same number
        self.reset_random_seed(seed_A)
        #roi_idx = 28 # get MxIF one hot segmentation mask roi index
        roi_idx = 3 # should be index for Lucas' segmentation mask
        A_label = A_img.crop((roi_idx * self.patch_size, 0, roi_idx * self.patch_size + self.patch_size, self.patch_size))
        #SHUNXING
        Atruth = self.transform_label(A_label)
        #print('%s ##### ' % torch.unique(Atruth))
        # the labels will be convert to tensor, use '* 255' to make the label value back to one hot label
        Atruth = Atruth * 255
        #print('%s ##### ' % torch.unique(Atruth))
        Atruth[Atruth==2] = 1 # for testing MxIF labels. It should not matter Lucas' 0-1 label. 
        #print('%s ##### ' % torch.unique(Atruth))


        A_tensor = None # the size of Tensor should match
        for i in range(0, len(tensor_A_list)):
            if A_tensor is None:
                A_tensor = tensor_A_list[i]
            else:
                A_tensor = torch.cat((A_tensor, tensor_A_list[i]), 0)
    
        return {'A': A_tensor, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'Atruth': Atruth}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def reset_random_seed(self, random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        random.seed(random_seed)
        np.random.seed(random_seed)
