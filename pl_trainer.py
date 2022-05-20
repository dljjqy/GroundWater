import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from models import *

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
        x, f = batch
        u = self(x)
        with torch.no_grad():
            v = self.rhs(u, f)
        loss = self.loss(u-v, torch.zeros_like(u-v))
        self.log('Train Los     s', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, f = batch
        u = self(x)

        v = self.rhs(u, f)
        diff = u-v
        loss = self.loss(diff, torch.zeros_like(diff))
        l1_loss = F.l1_loss(diff, torch.zeros_like(diff))
        self.log('Val Loss', loss)
        self.log('Val L1 Loss', l1_loss)

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


class FiniteDiffModuleSoft(pl.LightningModule):
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
        x, f = batch
        u = self(x)
        rhs = self.rhs(u, f)

        loss = self.loss(rhs, torch.zeros_like(rhs))
        self.log('Train Loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, f = batch
        u = self(x)
        rhs = self.rhs(u, f)
        
        loss = self.loss(rhs, torch.zeros_like(rhs))
        l1_loss = F.l1_loss(rhs, torch.zeros_like(rhs))
        self.log('Val Loss', loss)
        self.log('Val L1 Loss', l1_loss)

        u, rhs = u.squeeze().cpu().numpy(), rhs.squeeze().cpu().numpy()
        cap = str(f.max().item())
        np.save(self.fig_save_path + 'Val_u_' + cap, u)
        np.save(self.fig_save_path + 'Val_diff_' + cap, rhs)

        return {'Val Loss': loss, 'Val L1 Loss':l1_loss}

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr, max_iter=20)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        return [optimizer], [lr_scheduler]
