import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

class TanksAndTemples(Dataset):
    def __init__(self, config):
        self.subset = config.subset
        if self.others.subset == 'train':
            self.data_root = config.TRAIN_PATH
        if self.others.subset == 'test':
            self.data_root = config.TEST_PATH

        data = pd.read_csv(self.data_root)
        self.pcd_paths = data.iloc[:, 0].tolist()
        self.img_paths = data.iloc[:, 1].tolist()   
        self.npoints = config.N_POINTS

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
        pcd = np.load(pcd_path)
        pcd = self.pc_norm(pcd)
        pcd = torch.from_numpy(pcd).float()
        # return pcd, image
        return pcd