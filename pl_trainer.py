import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from models import *
from data_module import np2torch

class LinalgTrainer(pl.LightningModule):
    def __init__(self, loss, net, val_save_path='./u/lin/', A='./A.npz', lr=1e-3):
        super().__init__()
        self.loss = loss
        self.net = net
        self.val_save_path = val_save_path
        self.lr = lr
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.A = np2torch(A).to(device)
        # self.A = self.A.transpose()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, b, u = batch
        b = torch.transpose(b, 0, 1)
        y = self(x)
        mean_l2_loss = F.mse_loss(y.squeeze(), u)

        y = torch.transpose(torch.flatten(y, 2, 3).squeeze(), 0, 1)       
        y = torch.sparse.mm(self.A, y)
        linalg_loss = self.loss(y, b)
       
        self.log('Train linalg Loss', linalg_loss)
        self.log('Train Real Loss', mean_l2_loss)
        return {'loss' : linalg_loss}

    def validation_step(self, batch, batch_idx):
        x, b, u = batch
        b = torch.transpose(b, 0, 1)
        y = self(x)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!', y.shape)
        pre = y.squeeze().cpu().numpy()
        mean_l2_loss = F.mse_loss(y.squeeze(), u)

        y = torch.flatten(y, 2, 3).squeeze()[..., None]
        y = torch.sparse.mm(self.A, y)
        linalg_loss = self.loss(y, b)
        
        self.log('Val linalg Loss', linalg_loss)
        self.log('Val Real Loss', mean_l2_loss)
    
        np.save(self.val_save_path + str(batch_idx), pre)

        return {'Val linalg Loss': linalg_loss, 'Val Real Loss': mean_l2_loss}
    
    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr, max_iter=20)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        return [optimizer], [lr_scheduler]

class Hard_FD_Rec_Module(pl.LightningModule):
    def __init__(self, loss, net, fig_save_path, rhs, lr=1e-3):
        super().__init__()
        self.loss = loss
        self.net = net
        self.fig_save_path = fig_save_path
        self.rhs = rhs
        self.lr = lr

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # x, f = batch
        x, f, ans = batch

        u = self(x)
        with torch.no_grad():
            v = self.rhs(u, f)
            
        loss = self.loss(u-v, torch.zeros_like(u-v))
        loss0 = F.mse_loss(u.squeeze(), ans)

        self.log('Train Loss', loss)
        self.log('Real Loss', loss0)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, f, ans = batch
        u = self(x)

        v = self.rhs(u, f)
        diff = u-v
        
        loss = self.loss(diff, torch.zeros_like(diff))
        l1_loss = F.l1_loss(diff, torch.zeros_like(diff))
        loss0 = F.mse_loss(u.squeeze(), ans.squeeze())

        self.log('Val Loss', loss)
        self.log('Val L1 Loss', l1_loss)
        self.log('Val Real Loss', loss0)

        u, diff = u.squeeze().cpu().numpy(), diff.squeeze().cpu().numpy()
        cap = f'{f.max().item():.2f}'
        np.save(self.fig_save_path + 'Test_u_' + cap, u)
        np.save(self.fig_save_path + 'Test_diff_' + cap, diff)

        return {'Val Loss': loss, 'Val L1 Loss':l1_loss}
        
    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr, max_iter=20)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        return [optimizer], [lr_scheduler]

