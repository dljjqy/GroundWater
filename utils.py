from numpy import source
import torch 
import torch.nn.functional as F
from scipy.stats import multivariate_normal
from data_module import SquareDomain_fd
import numpy as np

class Dir_RHS_Rec(torch.nn.Module):
    '''
    Soft and Hard.
    First pad the Dirichlet boundary value,and then do the convolution to compute the
    finite difference equation.At the same time pad the result of convolution with the 
    Dirichlet value.
    Thus the u - v on the Dirichlet boundary is soft and interior is hard.
    '''
    def __init__(self, k, h, value, in_channels=1,
                    weight=[[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]]):
        super().__init__()
        # self.n = N
        self.k = k
        self.h = h
        self.value = value
        self.weight = torch.Tensor(weight).view(1, 1, len(weight), len(weight)).repeat(1, in_channels, 1, 1)
        
    def forward(self, u, f):
        self.weight = self.weight.type_as(u)

        v = u[..., 1:-1, 1:-1].detach()
        v = F.pad(v, (1,1,1,1), mode='constant', value=self.value)
        v[..., 1:-1, 1:-1] = torch.conv2d(v, self.weight, bias=None, stride=1, padding=0)

        cof = - self.h**2 / (self.k * 4)
        source = (cof * f)[..., 1:-1, 1:-1].detach()
        
        v[..., 1:-1, 1:-1] += source
        return v 

class XNeuYDir_RHS_Rec(torch.nn.Module):
    def __init__(self, k, h, dv, nv, 
                    in_channels=1, weight=[[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]]):
        super().__init__()
        self.k = k
        self.h = h
        self.dv = dv
        self.nv = nv
        self.weight = torch.Tensor(weight).view(1, 1, len(weight), len(weight)).repeat(1, in_channels, 1, 1)
        
    def forward(self, u, f):
        self.weight = self.weight.type_as(u)

        if self.nv == 0:
            v = u[..., 1:-1, :].detach()
            v = F.pad(v, (0,0,1,1), mode='constant', value=self.dv)
            v = F.pad(v, (1,1,0,0), mode='reflect')
            # print(v.shape)
        else:
            b, c, x, y = u.shape
            v = torch.empty((b, c, x, y+2))
            v[..., 1:-1, 1:-1] = u[..., 1:-1, :]
            v[..., 0, :] = self.dv
            v[..., -1, :] = self.dv

            ghost_value_left = u[..., :, 0] + 2*self.h * self.nv / self.k
            ghost_value_right = u[..., :, -1]+ 2*self.h * self.nv / self.k

            v[..., :, 0] = ghost_value_left
            v[..., :, -1] = ghost_value_right
        
        v[..., 1:-1, 1:-1] = torch.conv2d(v, self.weight, bias=None, stride=1, padding=0)
        cof = -self.h**2 / (self.k*4)
        source = (cof * f)[..., 1:-1, :].detach()

        v[..., 1:-1, 1:-1] += source
        return v[..., :, 1:-1]

class Dir_RHS(torch.nn.Module):
    def __init__(self, k, h, value, in_channels=1,
                    weight=[[0, 1, 0], [1, -4, 1], [0, 1, 0]]):
        super().__init__()
        self.k = k
        self.h = h
        self.value = value
        self.weight = torch.Tensor(weight).view(1, 1, len(weight), len(weight)).repeat(1, in_channels, 1, 1)
        
    def forward(self, u, f):
        self.weight = self.weight.type_as(u)
        self.weight.requires_grad = False

        v = torch.ones_like(u[..., 1:-1, 1:-1])
        # Assign 0 to the edge of u 
        w = F.pad(v, (1,1,1,1), mode='constant', value=0)
        u *= w
        # Assign Dirichlet to the edge of u 
        v = torch.zeros_like(u[..., 1:-1, 1:-1])
        v = F.pad(v, (1,1,1,1), mode='constant', value=self.value)
        u += v
        # Finite difference
        conv = torch.conv2d(u, self.weight, bias=None, stride=1, padding=0)
        cof = self.h**2 / self.k 
        source = (cof * f)[..., 1:-1, 1:-1].detach()

        rhs = conv - source
        rhs = F.pad(rhs, (1,1,1,1))
        return rhs

# class Dir_soft_RHS(object):
#     def __init__(self, k, h, value, in_channels=1,
#                     weight=[[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]]):
#         self.k = k
#         self.h = h
#         self.value = value
#         self.weight = torch.Tensor(weight).view(1, 1, len(weight), len(weight)).repeat(1, in_channels, 1, 1)
        
#     def forward(self, x, f):
#         u = x[..., 1:-1, 1:-1].detach()
#         u = F.pad(u, (1,1,1,1), mode='constant', value=self.value)
#         rhs_bc = u-x
#         conv = torch.conv2d(u, self.weight, bias=None, stride=1, padding=0)
#         source = (self.h * self.h * self.k * f / 4)[..., 1:-1, 1:-1].detach()
#         rhs_bc[...,1:-1, 1:-1] = (conv + source)
#         return rhs_bc

# class XNeuYDir_RHS(object):
#     def __init__(self, k, h, dvalue, nvalue, in_channels=1,
#                     weight=[[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]]):
#         # self.n = N
#         self.k = k
#         self.h = h
#         self.dvalue = dvalue
#         self.nvalue = nvalue
#         self.weight = torch.Tensor(weight).view(1, 1, len(weight), len(weight)).repeat(1, in_channels, 1, 1)

#     def forward(self, x, f):
#         u = x[..., 1:-1,:].detach()
#         u = F.pad(u, (0,0,1,1), mode='constant', value=self.dvalue)

def test_g(cap, a, h, ckpt_path, model, mean, sigma):
    sdm = SquareDomain_fd(
        left = -a, right = a,
        bottom = -a, top = a,
        dx = h, dy = h)

    X, Y = sdm.coordinate[0,:,:], sdm.coordinate[1,:,:]
    pos = np.empty(X.shape + (2,))
    pos[:,:,0] = X
    pos[:,:,1] = Y
    rv = multivariate_normal(mean, sigma)
    pd = cap * rv.pdf(pos)
    pd = pd[np.newaxis, ...]

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    model.freeze()

    # print(pd.shape)
    input = np.concatenate([sdm.coordinate, pd], axis=0)[np.newaxis, ...]
    input = torch.Tensor(input)
    # print(input.shape)

    return model(input).squeeze() 

if __name__ == '__main__':
    from models import UNet, WeightedLoss
    from pl_trainer import *
    from utils import *
    cap = -1
    pre = test_g(cap=float(cap), a=1, h=0.005,
                ckpt_path= './lightning_logs/version_0/checkpoints/epoch=16-step=16829.ckpt',
                 mean=[0,0], sigma = [[1e-4,0],[0,1e-4]],
                model=Hard_FD_Rec_Module( net = UNet(),
                                    loss = WeightedLoss(diff_fun = F.l1_loss),
                                    fig_save_path = './u/case1/',
                                    lr = 1e-3,
                                    rhs = Dir_RHS_Rec(k=1, h=0.005, value=0))).numpy()
                                    