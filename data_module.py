from matplotlib.pyplot import axis
import numpy as np
import torch
import pytorch_lightning as pl
from scipy import sparse
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from random import sample
from scipy.stats import multivariate_normal
from tqdm import tqdm


def np2torch(A_path):
    A = sparse.load_npz(A_path)
    # A = A.todense()
    values = A.data
    indices = np.vstack((A.row, A.col))
    
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(torch.float32)
    # return torch.from_numpy(A).to(torch.float32)
class LinalgDataset(Dataset):
    def __init__(self, path_b, path_u, path_f, N=128, a=1, mode='fit'):

        super().__init__()
        x = np.linspace(0, a, N)
        y = np.linspace(0, a, N)
        xx, yy = np.meshgrid(x, y)
        self.xx = xx[np.newaxis,...]
        self.yy = yy[np.newaxis,...]

        self.B = np.load(path_b)
        self.U = np.load(path_u)
        self.F = np.load(path_f)

    def __len__(self):
        return self.B.shape[0]

    def __getitem__(self, idx):
        b = self.B[idx, :]
        u = self.U[idx, :].reshape(self.xx.shape[1:])
        f = self.F[idx, :].reshape(self.xx.shape)
        data = np.concatenate((self.xx, self.yy, f), axis=0)

        data = torch.from_numpy(data).to(torch.float32)
        b = torch.from_numpy(b).to(torch.float32)
        u = torch.from_numpy(u).to(torch.float32)

        return data, b, u

class LinalgDataModule(pl.LightningDataModule):
    def __init__(self, train_b, train_u, train_f, val_b, val_u, val_f, N=128, batch_size=4):
        super().__init__()
        self.batch_size = batch_size
        self.B = train_b
        self.U = train_u
        self.F = train_f
        self.valB = val_b
        self.valU = val_u
        self.valF = val_f
        self.N = N

    def setup(self, stage):    
        if stage == 'fit' or stage is None:
            self.train_dataset = LinalgDataset(self.B, self.U, self.F, self.N, a=1)
            self.val_dataset = LinalgDataset(self.valB, self.valU, self.valF, self.N, a=1)
        if stage == 'test':
            pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                        shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, 
                        shuffle=False, num_workers=6)
    
    def test_dataloader(self):
        pass    

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

class SinDataGen(object):
    def __init__(self, domain, f, Q):
        self.Q = sample(Q, len(Q))
        self.xx = domain.coordinate[0].squeeze()
        self.yy = domain.coordinate[1].squeeze()
        self.f = f

    def save2dir(self, dir):
        for q in tqdm(self.Q):
            force = q * self.f(self.xx, self.yy)
            data = np.concatenate((self.xx[np.newaxis, ...], self.yy[np.newaxis, ...], 
                                    force[np.newaxis, ...]), axis=0)
            name = dir + str(q)
            np.save(name, data)

class DataGen(object):
    def __init__(self, caps, domain, mu=[0, 0], sigma=[[1, 0], [0, 1]]):
        self.caps = sample(caps, len(caps))
        self.mu = np.array(mu)
        self.domain = domain
        self.sigma = np.array(sigma)

    def _gauss(self, cap):
        X, Y = self.domain.coordinate[0,:,:], self.domain.coordinate[1,:,:]
        pos = np.empty(X.shape + (2,))
        pos[:,:,0] = X
        pos[:,:,1] = Y
        rv = multivariate_normal(self.mu, self.sigma)
        pd = rv.pdf(pos)
        return cap * pd

    def layout(self, cap):
        layout = np.zeros_like(self.domain.coordinate[0,:,:])
        Nx = round(self.mu[0] - self.domain.left/self.domain.dx)
        Ny = round(self.mu[1] - self.domain.right/self.domain.dy)
        layout[Nx, Ny] += cap
        return layout[np.newaxis,...]


    def save2dir(self, dir):
        for idx in tqdm(range(len(self.caps))):
            f = self._gauss(self.caps[idx])
            f = f[np.newaxis, ...]
            layout = self.layout(self.caps[idx])
            layout3d = np.concatenate((self.domain.coordinate, layout, f), axis=0)
            np.save(Path(dir + 'gauss_' + str(idx)), layout3d)
        return True

class DataGenGauss(object):
    def __init__(self, caps, domain, mu=[0, 0], sigma=[[1, 0], [0, 1]]):
        self.caps = sample(caps, len(caps))
        self.mu = np.array(mu)
        self.domain = domain
        self.sigma = np.array(sigma)

    def _gauss(self, cap):
        X, Y = self.domain.coordinate[0,:,:], self.domain.coordinate[1,:,:]
        pos = np.empty(X.shape + (2,))
        pos[:,:,0] = X
        pos[:,:,1] = Y
        rv = multivariate_normal(self.mu, self.sigma)
        pd = rv.pdf(pos)
        return cap * pd

    def save2dir(self, dir):
        for idx in tqdm(range(len(self.caps))):
            layout = self._gauss(self.caps[idx])
            layout = layout[np.newaxis, ...]
            
            layout3d = np.concatenate((self.domain.coordinate, layout), axis=0)
            np.save(Path(dir + 'gauss_' + str(idx)), layout3d)
        return True

class FGauess(object):
    def __init__(self, caps, domain, mus, sigma, N):
        self.domain = domain
        self.mus = mus
        self.sigma = sigma
        self.N = N
        self.caps = caps

    def _gauss(self, caps):
        X, Y = self.domain.coordinate[0,:,:], self.domain.coordinate[1,:,:]
        pos = np.empty(X.shape + (2,))
        pos[:,:,0] = X
        pos[:,:,1] = Y
        pd = np.zeros_like(X)
        for i in range(4):
            rv = multivariate_normal(self.mus[i], self.sigma)
            pd = pd + caps[i] * rv.pdf(pos)
        return pd  

    def save2dir(self, dir):
        for i in tqdm(range(self.N)):
            cs = sample(self.caps, 4)
            layout = self._gauss(cs)
            layout = layout[np.newaxis, ...]
            layout3d = np.concatenate((self.domain.coordinate, layout), axis=0)
            np.save(Path(dir + 'fgauss_' + str(i)), layout3d)
        return True

class ValDataset(Dataset):
    def __init__(self, path, domain=SquareDomain_fd(dx=5, dy=5,left=-500,right=500,top=500,bottom=-500), 
                            mu=[0,0], sigma = [[1.56,0],[0,1.56]]):
        super().__init__()
        self.path = list(Path(path).glob('*.npy'))
        self.mu = mu
        self.sigma = sigma
        self.domain = domain

    def __getitem__(self, index):
        Q = float(self.path[index].name[:-4])
        input = torch.from_numpy(self._gendata(Q)).to(dtype=torch.float32)
        label = torch.from_numpy(np.load(self.path[index]))
        return input, label

    def __len__(self):
        return len(self.path)

    def _gendata(self, Q):
        X, Y = self.domain.coordinate[0,:,:], self.domain.coordinate[1,:,:]
        f = np.zeros_like(X)
        Nx = round(self.mu[0] - self.domain.left/self.domain.dx)
        Ny = round(self.mu[1] - self.domain.right/self.domain.dy)
        f[Nx, Ny] += Q
        f = f[np.newaxis, ...]
        data = np.concatenate((self.domain.coordinate, f), axis=0)
        return  data

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
        # layout = layout3d[None, 3, :].detach()
        # layout3d = layout3d[None, :3, :].detach()

        if self.trans:
            layout3d = self.trans(layout3d)
        return layout3d, layout

    def __len__(self):
        return len(self.data)

class SinDataSet(Dataset):
    def __init__(self, Q, f, u, h=0.01, a=1, mode='fit'):
        super().__init__()
        self.f = f
        self.u = u
        self.Q = Q
        
        N = round(a/h)
        x = np.linspace(0, a, N)
        y = np.linspace(0, a, N)
        self.xx, self.yy = np.meshgrid(x, y)

    def __getitem__(self, idx):
        force = self.Q[idx] * self.f(self.xx, self.yy)
        ans = self.Q[idx] * self.u(self.xx, self.yy)

        data = np.concatenate((self.xx[np.newaxis,...], self.yy[np.newaxis,...], force[np.newaxis, ...]), axis=0)

        rhs = torch.from_numpy(force)[None, ...]
        data = torch.from_numpy(data)
        label = torch.from_numpy(ans)

        return data.to(torch.float32), rhs.to(torch.float32), label.to(torch.float32)

    def __len__(self):
        return len(self.Q)

class SinDataModule(pl.LightningDataModule):
    def __init__(self, h, rate=0.99 ,a=2, N=2000, batch_size=4):
        super().__init__()
        self.Q = sample(list(np.linspace(0.5, a, N)), N)
        self.batch_size = batch_size
        self.rate = rate
        self.f = lambda x,y: 8*np.pi*np.pi*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
        self.u = lambda x,y: np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
        self.h = h

    def setup(self, stage):
        n = round(self.rate * len(self.Q))    
        if stage == 'fit' or stage is None:
            self.train_dataset = SinDataSet(self.Q[:n], self.f, self.u, self.h)
            self.val_dataset = SinDataSet(self.Q[n:], self.f, self.u, self.h)
        if stage == 'test':
            pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                        shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, 
                        shuffle=False, num_workers=6)
    
    def test_dataloader(self):
        pass       

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
            # self.val_dataset = ValDataset('./test_data/water/')

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
    ds = LinalgDataset('./B2nd.npy', 'U2nd.npy', h=0.01)

    data, b, u = ds[0]
    print(data.shape)
    print(b.shape)
    print(u.shape)



# Sin Data Gen:
    # def f(xx, yy):
    #     return 8 * np.pi * np.pi * np.sin(2*np.pi*xx) + np.sin(2*np.pi*yy)
    # root = "./data/sin/"
    # domain = SquareDomain_fd(dx=0.01, dy=0.01, left=0, right=1, top=1, bottom=0)
    # Q = list(np.arange(0.001, 2.001, 0.001))
    # datagenerator = SinDataGen(domain, f, Q)
    # datagenerator.save2dir(root)
    # dataset = LayoutDataset("./data/green/")
    # data, f = dataset[0]
    # print(data.shape)
    # print(f.shape)
#   Gauss Data Generator case1
    # High resolution of Green function dataset.
    # domain = SquareDomain_fd(    
    #         dx=5, 
    #         dy=5,
    #         left=-500,
    #         right=500,
    #         top=500,
    #         bottom=-500)
    # caps = list(np.arange(5000, 15000, 5))
    # # caps = sample(caps, len(caps))
    # mu = [0.0, 0.0]
    # sigma = [[1.56,0],[0,1.56]]
    # singleWellDataGenerator = DataGen(caps, domain, mu, sigma)
    # singleWellDataGenerator.save2dir("./data/water200/")

#   Four pumps green data 
    # domain = SquareDomain_fd(dx=0.005, dy=0.005,left=-1,right=1,top=1,bottom=-1)
    # caps = list(np.arange(0, 2, 0.001))
    # mus = ([0.5, 0.5], [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5])
    # sigma = [[1e-4,0],[0,1e-4]]
    # N = 5000
    # FourWellsDataGenerator = FGauess(caps, domain, mus, sigma, N)
    # FourWellsDataGenerator.save2dir("./data/four/")

    # Water Data:
    # domain = SquareDomain_fd(    
    #         dx=2.5, 
    #         dy=2.5,
    #         left=-500,
    #         right=500,
    #         top=500,
    #         bottom=-500)
    # caps = list(np.arange(0, 20000, 10))
    # caps = sample(caps, len(caps))
    # mu = [0.0, 0.0]
    # sigma = [[1.56,0],[0,1.56]]
    # singleWellDataGenerator = DataGenGauss(caps, domain, mu, sigma)
    # singleWellDataGenerator.save2dir("./data/case2/")