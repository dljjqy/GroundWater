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

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, u, b = batch
        y = self(x)
        mean_l2_loss = F.mse_loss(y.squeeze(), u)

        bc = y.size(0)
        y = y.view((bc, -1, 1))
        linalg_loss = 0
        for idx in range(bc):
            linalg_loss += self.loss(torch.sparse.mm(self.A, y[idx]).squeeze(), b[idx])

        self.log('Train linalg Loss', linalg_loss/bc)
        self.log('Train Real Loss', mean_l2_loss)
        return {'loss' : mean_l2_loss}

    def validation_step(self, batch, batch_idx):
        x, u, b = batch
        y = self(x)

        # print(y.shape, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        mean_l2_loss = F.mse_loss(y.squeeze(), u[0])

        y = y.view(-1, 1)
        linalg_loss = self.loss(torch.sparse.mm(self.A, y), b.view(-1, 1))
        self.log('Val linalg Loss', linalg_loss)
        self.log('Val Real Loss', mean_l2_loss)
        
        y = y.squeeze().cpu().numpy()
        np.save(self.val_save_path + str(batch_idx), y)

        return {'Val linalg Loss': linalg_loss/4, 'Val Real Loss': mean_l2_loss}
    
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

