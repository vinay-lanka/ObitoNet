import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import cv2

class TanksAndTemples(Dataset):
    def __init__(self, config):
        print(config)
        self.subset = config.others.subset
        if self.subset == 'train':
            self.data_root = config._base_.TRAIN_PATH
        if self.subset == 'test':
            self.data_root = config._base_.TEST_PATH

        data = pd.read_csv(self.data_root)
        self.pcd_paths = data.iloc[:, 0].tolist()
        self.img_paths = data.iloc[:, 1].tolist()   
        self.npoints = config._base_.N_POINTS
        self.pc_path_root = config._base_.PC_PATH

    def __len__(self):
        return len(self.pcd_paths)
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        pcd_path = self.pcd_paths[idx]
        # img_path = self.img_paths[idx]
        # image = cv2.imread(os.path.join(self.pc_path_root, img_path))
        # image = cv2.resize(image, (256, 256))
        pcd = np.load(os.path.join(self.pc_path_root, pcd_path))
        pcd = self.pc_norm(pcd)
        pcd = torch.from_numpy(pcd).float()
        # return pcd, image
        return pcd