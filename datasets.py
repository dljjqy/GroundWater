import torch
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
# from utils import *

class MyDataset(Dataset):
    def __init__(self, pathB, pathF):
        super().__init__()
        self.B = np.load(pathB)
        self.F = np.load(pathF)

    def __len__(self):
        return self.B.shape[0]

    def __getitem__(self, idx):
        data = self.F[idx, :]
        b = self.B[idx, :]

        data = torch.from_numpy(data).to(torch.float32)
        b = torch.from_numpy(b).to(torch.float32)
        return  data, b

class MyDataModule(pl.LightningDataModule):
    
    def __init__(self, data_path, batch_size, order=2, mode='F'):
        super().__init__()
        self.trainF = data_path + mode + '.npy'
        self.valF = data_path + 'Val' + mode + '.npy'

        if order == 2:
            self.trainB = data_path + 'B2nd.npy'
            self.valB = data_path + 'ValB2nd.npy'
        elif order == 4:
            self.trainB = data_path + 'B4th.npy'
            self.valB = data_path + 'ValB4th.npy'
        self.batch_size = batch_size

    def setup(self, stage):    
        if stage == 'fit' or stage is None:
            self.train_dataset = MyDataset(self.trainB, self.trainF)
            self.val_dataset = MyDataset(self.valB, self.valF)
        
        if stage == 'test':
            pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=6)
    
    def test_dataloader(self):
        pass  
