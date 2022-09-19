import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lbfgsnew import LBFGSNew
from scipy import sparse
from matplotlib import cm

def np2torch(A_path):
    A = sparse.load_npz(A_path)
    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape
    del A
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(torch.float32)

def pad_neu_bc(x, h, pad=(1, 1, 0, 0), g = 0):
    val = 2 * h * g
    x = F.pad(x, pad=pad, mode='reflect')
    
    if pad == (1, 1, 0, 0):
        x[..., :, 0] -= val
        x[..., :,-1] -= val
    elif pad == (0, 0, 1, 1):
        x[..., 0, :] -= val
        x[...,-1, :] -= val
    return x

def pad_diri_bc(x, pad=(0, 0, 1, 1), g = 0):
    x = F.pad(x, pad=pad, mode='constant', value=g)
    return x

def conv_rhs(x):
    kernel = torch.tensor([[[[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]]])
    kernel = kernel.type_as(x)
    return F.conv2d(x, kernel)

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

def diri_rhs(x, f, h, g=0):
    '''
    All boundaries are Dirichlet type.
    Netwotk should output prediction without boundary points.(N-2 x N-2)
    '''
    x = pad_diri_bc(x, pad=(1, 1, 1, 1), g=g)
    rhs = conv_rhs(x)
    return rhs + h*h*f[..., 1:-1, 1:-1]

def neu_rhs(x, f, h, g_n=0, g_d=0):
    '''
    Left,right boundary are neumann type.
    Top,bottom boundary are dirichlet type.
    Network should output prediction with boundary points.(N x N)
    '''
    x = pad_neu_bc(x, h, pad=(1, 1, 0, 0), g=g_n)
    x = pad_diri_bc(x, (0, 0, 1, 1), g=g_d)
    rhs = conv_rhs(x)
    return rhs + h*h*f[...,1:-1, 1:-1]


class pl_conv_model(pl.LightningModule):
    def __init__(self, loss, net, rhs, lr, N, a):
        super().__init__()
        self.loss = loss
        self.net = net
        self.lr = lr
        self.rhs = rhs
        self.h = 2*a / (N - 1)

        x = np.linspace(-a, a, N)
        y = np.linspace(-a, a, N)
        self.x, self.y = np.meshgrid(x, y)
        self.x = self.x[1:-1, 1:-1]
        self.y = self.y[1:-1, 1:-1]


        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.ax2 = self.fig.add_subplot(1, 2, 2)
        
        del x, y

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        f = x[:, -1:, ...]
        # print(f.shape)
        u = self(x)
        with torch.no_grad():
            rhs = self.rhs(u, f, self.h)
        
        loss = self.loss(u, rhs)
        self.log('loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        f = x[:, -1:, ...]

        u = self(x)
        rhs = self.rhs(u, f, self.h)
        loss = self.loss(u, rhs)
        diff = torch.abs(u-rhs)
        if self.current_epoch % 5 == 0:
            u = u.cpu().numpy().reshape(self.x.shape)
            diff = diff.cpu().numpy().reshape(self.x.shape)
            surf = self.ax1.plot_surface(self.x, self.y, u, cmap=cm.coolwarm)
            im = self.ax2.imshow(diff, cmap='jet')
            tensorboard = self.logger.experiment
            tensorboard.add_figure(tag = 'surf&diff', figure=self.fig, global_step=self.current_epoch)
        
        self.log('val loss', loss)
        return {'val loss': loss}

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        return [optimizer], [lr_scheduler]

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
            # energy_loss = energy(y, self.A, b)
            jacobian_loss = self.loss(y, rhs)
            mse_linalg_loss = mse_loss(y, self.A, b)
        energy_loss = energy(y, self.A, b)
        

        parameters = [p for p in self.net.parameters() if p.grad is not None and p.requires_grad]
        if len(parameters) == 0:
            total_norm = 0.0
        else:
            device = parameters[0].grad.device
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2.0).item()

        self.log_dict({
            'Jacobian Iteration l1 Loss': jacobian_loss,
            'Mean Energy Loss':energy_loss,
            'MSE Linalg Loss':mse_linalg_loss,
            'Grad Norm':total_norm
        })

        # return {'loss' : jacobian_loss}
        return {'loss' : energy_loss}

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