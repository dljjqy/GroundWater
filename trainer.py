import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from datasets import np2torch
from pathlib import Path
from lbfgsnew import LBFGSNew
def gradient_descent(x, A, b):
    r = matrix_batched_vectors_multipy(A, x) - b
    Ar = matrix_batched_vectors_multipy(A, r)
    alpha = batched_vec_inner(r, r)/batched_vec_inner(r, Ar)
    y = x + alpha * r
    return y

def matrix_batched_vectors_multipy(A, y):
    """
    Sparse matrix multiply Batched vectors
    """
    y = torch.transpose(y, 0, 1)
    v = torch.sparse.mm(A, y)
    v = torch.transpose(v, 0, 1)
    return v

def batched_vec_inner(x, y):
    """
    inner product of Batched vectors x and y
    """
    b, n = x.shape
    return torch.bmm(x.view((b, 1, n)), y.view((b, n, 1))) 

def energy(x, A, b):
    Ax = matrix_batched_vectors_multipy(A, x)
    bx = batched_vec_inner(b, x)
    xAx = batched_vec_inner(x, Ax)
    return (xAx/2 - bx).mean()

def mse_loss(x, A, b):
    Ax = matrix_batched_vectors_multipy(A, x)
    return F.mse_loss(Ax, b)


class pl_Model(pl.LightningModule):
    def __init__(self, loss, net, val_save_path='./u/', data_path='./data/', lr=1e-2, order=2):
        self.val_save_path = Path(val_save_path)
        super().__init__()
        self.loss = loss
        self.net = net
        self.lr = lr
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if order == 2:
            self.D = np2torch(data_path+'D2nd.npz').to(device)
            self.A = np2torch(data_path+'A2nd.npz').to(device)
            self.order = 4
        elif order == 4:
            self.D = np2torch(data_path+'D4th.npz').to(device)
            self.A = np2torch(data_path+'A4th.npz').to(device)
            self.order = 20

        if not self.val_save_path.is_dir():
            self.val_save_path.mkdir(exist_ok=False)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, b = batch
        y = self(x)
        y = y.flatten(1, 3)
        with torch.no_grad():
            rhs = self.rhs(y, b)
            energy_loss = energy(y, self.A, b)
            mse_linalg_loss = mse_loss(y, self.A, b)
        jacobian_loss = self.loss(y, rhs)

        self.log_dict({
            'Jacobian Iteration l1 Loss': jacobian_loss,
            'Mean Energy Loss':energy_loss,
            'MSE Linalg Loss':mse_linalg_loss
        })
        return {'loss' : jacobian_loss}

    def validation_step(self, batch, batch_idx):
        x, b = batch
        y = self(x)
        np.save(f'{self.val_save_path}/{batch_idx}', y.squeeze().cpu().numpy())

        y = torch.flatten(y, 1, 3)
        rhs = self.rhs(y, b)
        energy_loss = energy(y, self.A, b)
        mse_linalg_loss = mse_loss(y, self.A, b)
        jacobian_loss = self.loss(y, rhs)

        self.log_dict({
            'Val Jacobian Iteration l1 Loss': jacobian_loss,
            'Val Mean Energy Loss':energy_loss,
            'Val MSE Linalg Loss':mse_linalg_loss
        })
        return {
            'Val Jacobian Iteration l1 Loss': jacobian_loss,
            'Val Mean Energy Loss':energy_loss,
            'Val MSE Linalg Loss':mse_linalg_loss}
    
    def rhs(self, y, b):
        rhs = matrix_batched_vectors_multipy(self.D, y)
        rhs = b/self.order - rhs 
        return rhs

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        return [optimizer], [lr_scheduler]


class pl_lbfgs_Model(pl_Model):
    def __init__(self, *args, **kwargs):
        super(pl_lbfgs_Model, self).__init__(*args, **kwargs)
        self.automatic_optimization = False
    
    def training_step(self, batch, batch_idx):
        x, b = batch
        opt = self.optimizers()
        def closure():
            if torch.is_grad_enabled():
                opt.zero_grad()
            y = self(x).flatten(1, 3)
            with torch.no_grad():
                rhs = self.rhs(y, b)
            loss = self.loss(y, rhs)
            if loss.requires_grad:
                self.manual_backward(loss)
            return loss
        opt.step(closure)

        y = self(x).flatten(1, 3)
        with torch.no_grad():
            rhs = self.rhs(y, b)
            energy_loss = energy(y, self.A, b)
            mse_linalg_loss = mse_loss(y, self.A, b)
        jacobian_loss = self.loss(y, rhs)

        self.log_dict({
            'Jacobian Iteration l1 Loss': jacobian_loss,
            'Mean Energy Loss':energy_loss,
            'MSE Linalg Loss':mse_linalg_loss
        })

    def configure_optimizers(self):
        optimizer = LBFGSNew(self.parameters(), 
            history_size=7, max_iter=2, line_search_fn=True, batch_mode=True)
        return [optimizer]