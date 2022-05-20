import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from random import sample
from scipy.stats import multivariate_normal


class SquareDomain_fd(object):
    def __init__(self, dx, dy, left, right, top, bottom):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.dx = dx
        self.dy = dy

        Nx = round((self.right - self.left)/dx)
        # print(Nx)
        Ny = round((self.top - self.bottom)/dx)

        x = np.linspace(self.left, self.right, Nx)
        y = np.linspace(self.bottom, self.top, Ny)
        xx, yy = np.meshgrid(x, y)
        xx = xx[np.newaxis, ...]
        yy = yy[np.newaxis, ...]

        self.coordinate = np.concatenate((xx, yy), axis=0)
        self.D = np.zeros((Nx, Ny))
    
    @property
    def shape(self):
        return self.D.shape

class DataGenGauss(object):
    def __init__(self, caps, domain, mu=[0, 0], sigma=[[1, 0], [0, 1]]):
        self.caps = sample(caps, len(caps))
        self.mu = np.array(mu)
        self.domain = domain
        self.data = []
        self.sigma = np.array(sigma)

    def _gauss(self, cap):
        X, Y = self.domain.coordinate[0,:,:], self.domain.coordinate[1,:,:]
        pos = np.empty(X.shape + (2,))
        pos[:,:,0] = X
        pos[:,:,1] = Y
        rv = multivariate_normal(self.mu, self.sigma)
        pd = rv.pdf(pos)
        return cap * pd
    
    
    def gen_data(self):
        for c in self.caps:
            layout = self._gauss(c)
            self.data.append(layout)
    
    def save2dir(self, dir):
        if len(self.data) < len(self.caps):
            self.data=[]
            self.gen_data()
        for idx, layout in enumerate(self.data):
            layout = layout[np.newaxis, ...]
            layout3d = np.concatenate((self.domain.coordinate, layout), axis=0)
            np.save(Path(dir + 'gauss_' + str(idx)), layout3d)
        return True


class LayoutDataset(Dataset):

    def __init__(self, path, trans=False, mode='fit'):
        super().__init__()
        self.path = Path(path)
        self.trans = trans
        data = list(self.path.glob('*.npy'))
        N_fit = round(len(data) * 0.99)
        N_val = round(len(data) * 0.01) + N_fit
        
        if mode ==  'fit' or mode is None:
            self.data = data[0: N_fit]
        elif mode == 'val':
            self.data = data[N_fit: N_val]
        elif mode == 'test':
            self.data = data[N_val:]
    
    def __getitem__(self, index):

        layout3d = (torch.from_numpy(np.load(self.data[index])).to(dtype=torch.float32))
        layout = layout3d[None, 2, :].detach()
        
        if self.trans:
            layout3d = self.trans(layout3d)
        return layout3d, layout

    def __len__(self):
        return len(self.data)

class DataModule(pl.LightningDataModule):
    
    def __init__(self, path, trans=False, batch_size=4):
        super().__init__()
        self.path = path
        self.trans = trans
        self.bath_size = batch_size
        
    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.train_dataset = LayoutDataset(self.path, self.trans, 'fit')
            self.val_dataset = LayoutDataset(self.path, self.trans, 'val')
        if stage == 'test':
            self.test_dataset = LayoutDataset(self.path, self.trans, 'test')
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.bath_size, 
                        shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, 
                        shuffle=False, num_workers=6)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1,
                         shuffle=False, num_workers=6)


class TestDataSet(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = list(Path(path).glob('*.npy'))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        layout3d = ((torch.from_numpy(np.load(self.data[index])).to(dtype=torch.float32)))
        layout = layout3d[None, 2, :].detach()
        return layout3d, layout

if __name__ == '__main__':
#   Gauss Data Generator case1
    # High resolution of Green function dataset.
    # domain = SquareDomain_fd(    
    #         dx=0.005, 
    #         dy=0.005,
    #         left=-1,
    #         right=1,
    #         top=1,
    #         bottom=-1)
    # caps = list(np.arange(0, 2, 0.001))
    # caps = sample(caps, len(caps))
    # mu = [0.0, 0.0]
    # sigma = [[1e-4,0],[0,1e-4]]
    # singleWellDataGenerator = DataGenGauss(caps, domain, mu, sigma)
    # singleWellDataGenerator.save2dir("./data/case1/")

    # Water Data:
    domain = SquareDomain_fd(    
            dx=2.5, 
            dy=2.5,
            left=-500,
            right=500,
            top=500,
            bottom=-500)
    caps = list(np.arange(0, 20000, 10))
    caps = sample(caps, len(caps))
    mu = [0.0, 0.0]
    sigma = [[1.56,0],[0,1.56]]
    singleWellDataGenerator = DataGenGauss(caps, domain, mu, sigma)
    singleWellDataGenerator.save2dir("./data/case2/")