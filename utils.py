import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve2d
from scipy.stats import multivariate_normal
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import MyDataModule
from models import model_names
from trainer import diri_rhs, neu_rhs, pl_Model, pl_conv_model, pl_lbfgs_Model
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

def normal(x, y, h, mean=[0, 0]):
    var = np.diag([1] * 2) * h**2 / np.sqrt(2 * np.pi)
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, var)
    return rv.pdf(pos)

def normal_fourth(x, y, h):
    m1 = [-0.5, -0.5]
    m2 = [0.5, 0.5]
    m3 = [-0.5, 0.5]
    m4 = [0.5, -0.5]
    var = np.diag([1] * 2) * h**2 / (2 * np.pi)
    pos = np.dstack((x, y))
    rv1 = multivariate_normal(m1, var)
    rv2 = multivariate_normal(m2, var)
    rv3 = multivariate_normal(m3, var)
    rv4 = multivariate_normal(m4, var)
    return rv1.pdf(pos) + rv2.pdf(pos) + rv3.pdf(pos) + rv4.pdf(pos) 
    
def R(x, y):
    return np.sqrt(x**2 + y**2)

def yita11_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask * ((6 * (3 - 4 * r)/np.pi)/H2)

def yita12_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return  mask * ((12 * (5 * r**2 - 8 * r + 3)/np.pi)/H2)

def yita22_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask*((12 * (15 * r**2 - 20 * r + 6)/np.pi)/H2)

def yita23_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask*((-60 * (7 * r**3 - 15 * r**2 + 10 * r - 2)/np.pi)/H2)

def yita25_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask*((84*(24*r**5-70*r**4+70*r**3-25*r**2+1)/np.pi)/H2)

def yita2cos_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask*(((-1 / (9*np.pi**4-104*np.pi**2+48)) * \
            ((81*np.pi/16)*(3*np.pi**4-32*np.pi**2+48)*np.cos(3*np.pi*r) +
                2*np.pi*(9*np.pi**4-80*np.pi**2+48) * np.cos(2*np.pi*r) + 
                ((np.pi/16)*(45*np.pi**2+32*np.pi**2-48)*np.cos(np.pi*r) + 144*np.pi)))/H2)

def fd_A_with_bc(n, order=2):
    N2 = n**2
    A = sparse.lil_matrix((N2, N2))
    if order == 2:
        for i in range(1, n-1):
            for j in range(1, n-1):
                idx = i * n + j
            
                A[idx, idx] += 4
                A[idx, idx-1] = -1
                A[idx, idx+1] = -1
                A[idx, idx-n] = -1
                A[idx, idx+n] = -1
    elif order == 4:
        for i in range(1, n-1):
            for j in range(1, n-1):
                idx = i * n + j
                A[idx, idx] = 20
                A[idx, idx-1] = -4
                A[idx, idx+1] = -4
                A[idx, idx-n] = -4
                A[idx, idx+n] = -4
                
                A[idx, idx-n-1] = -1
                A[idx, idx-n+1] = -1
                A[idx, idx+n-1] = -1
                A[idx, idx+n+1] = -1
                
    #Homogeneous Dirichlet Boundary
    for i in range(0, n):
        idx = 0 * n + i
        A[idx, idx] = 1
        
        idx = (n-1) * n + i
        A[idx, idx] = 1

        idx = i * n
        A[idx, idx] = 1
        
        idx = i * n + n - 1 
        A[idx, idx] = 1       
    A = A.tocoo()
    return A

def fd_b_bc(f, h, order=2):
    n, _ = f.shape
    h2 = h**2
    b = np.zeros(n**2)

    if order == 2:
        for i in range(1, n-1):
            for j in range(1, n-1):
                idx = i * n + j
                b[idx] = f[i, j]*h2
    elif order == 4:
        for i in range(1, n-1):
            for j in range(1, n-1):
                idx = i * n + j
                b[idx] += (8*f[i,j] + f[i-1,j] + f[i+1,j] + f[i,j-1] + f[i, j+1])/2*h2
    return b

def fd_A(num, order=2):
    n = num - 2
    N = n**2
    
    if order == 2:
        B = sparse.diags([4]*n) + sparse.diags([-1]*(n-1), -1) + sparse.diags([-1]*(n-1), 1)
        A = sparse.block_diag([B]*n)
        A += sparse.diags([-1]*(N-n), -n)
        A += sparse.diags([-1]*(N-n), n)
    
    elif order == 4:
        B = sparse.diags([20]*n) + sparse.diags([-4]*(n-1), 1) + sparse.diags([-4]*(n-1), -1)
        A = sparse.block_diag([B]*n)
        A = A + sparse.diags([-4]*(n*(n-1)), n) + sparse.diags([-4]*(n*(n-1)), -n)

        l = ([0] + [-1]*(n-1))*(n-1) + [0]
        A = A + sparse.diags(l, n-1) + sparse.diags(l, -n+1)
        
        l = (([-1]*(n-1) + [0]) * (n-1))[:-1]
        A = A + sparse.diags(l, n+1) + sparse.diags(l, -n-1)
    return A.tocoo()

def fd_b(f, h, order=2):
    n, _ = f.shape
    n = n - 2
    b = np.zeros(n**2)
    if order == 2:
        b += (f[1:-1, 1:-1].flatten())
        b *= (h**2)
    elif order == 4:
        kernel = np.array([[0, 1, 0], [1, 8, 1], [0, 1, 0]])
        b += convolve2d(f, kernel, mode='valid').flatten()[::-1]
        b *= (h**2/2)
    return b

def apply_dirichlet_bc(b, bc, g, order=2):
    '''
    bc --> True for soft, Flase for Hard
    g --> the value of boundary
    '''
    if g == 0:
        return b
    n = int(np.sqrt(len(b)))
    if bc:
        for i in range(0, n):
            idx = 0 * n + i
            b[idx] = g
            
            idx = (n-1) * n + i
            b[idx] = g

            idx = i * n
            b[idx] = g
            
            idx = i * n + n - 1 
            b[idx] = g 
    else:
        if order == 2:
            for i in range(0, n):
                idx = 0 * n + i
                b[idx] += g
                
                idx = (n-1) * n + i
                b[idx] += g

                idx = i * n
                b[idx] += g
                
                idx = i * n + n - 1 
                b[idx] += g

        elif order == 4:
            for i in range(0, n):
                idx = 0 * n + i
                b[idx] += 6*g
                
                idx = (n-1) * n + i
                b[idx] += 6*g

                idx = i * n
                b[idx] += 6*g
                
                idx = i * n + n - 1 
                b[idx] += 6*g
            b[0] -= g
            b[n-1] -= g
            b[(n-1)*n] -= g
            b[-1] -= g
    return b

def gen_hyper_dict(gridSize, batch_size, mode, net, features, four=False, neu=False,
                    label='jac', lr=1e-3, bc=False, max_epochs=80, ckpt=False, conv=False, a=1):
    exp_name = f'{mode:s}_{gridSize}_{net}_{features}_bs{batch_size}_{label}_lr{lr:.0e}'
    data_name = f'normal{gridSize}'
    
    if bc:
        exp_name += '_bc'
        data_name += '_bc'

    if four:
        exp_name = f'4{exp_name}'
        data_name = f'4{data_name}'
    
    data_path = f'./data/{data_name}/'
    in_c = 3 if mode == 'F' else 2

    model = model_names[net](in_c, 1, features, bc)
    if conv:
        exp_name = 'conv_' + exp_name

    if ckpt:
        exp_name = 'resume_' + exp_name

    dc = {'max_epochs':max_epochs, 'precision':32, 'check_val_every_n_epoch':1, 'ckpt_path':ckpt, 'mode':'fit', 'gpus':1}
    dc['pl_dataModule'] = MyDataModule(data_path, batch_size, 2, mode)
    # dc['check_point'] = ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', mode = 'min', every_n_train_steps = 0, 
    #                         every_n_epochs = 1, train_time_interval = None, save_top_k = 3, save_last = True,)
    dc['check_point'] = ModelCheckpoint(monitor = 'val loss', mode = 'min', every_n_train_steps = 0, 
                            every_n_epochs = 1, train_time_interval = None, save_top_k = 3, save_last = True,)
    
    
    dc['logger'] = TensorBoardLogger('./lightning_logs/', exp_name)
    
    if conv:
        if neu:
            dc['pl_model'] = pl_conv_model(F.l1_loss, model, neu_rhs, lr, gridSize, a)
        else:
            dc['pl_model'] = pl_conv_model(F.l1_loss, model, diri_rhs, lr, gridSize, a)

    else:
        dc['pl_model'] = pl_Model(F.l1_loss, model, f'./valu/{exp_name}/', data_path, lr, 2)

    if ckpt:
        parameters = torch.load(ckpt)
        dc['pl_model'].load_state_dict(parameters['state_dict'])

    return dc

def main(kwargs):
    # Initilize the Data Module
    dm = kwargs['pl_dataModule']
    
    # Initilize the model
    pl_model = kwargs['pl_model']
    # Initilize Pytorch lightning trainer   
    pl_trainer = pl.Trainer(
        gpus = kwargs['gpus'],
        callbacks = kwargs['check_point'],
        max_epochs = kwargs['max_epochs'],
        precision = kwargs['precision'],
        check_val_every_n_epoch = kwargs['check_val_every_n_epoch'],
        log_every_n_steps=10,
        logger = kwargs['logger'],
    )
    if kwargs['mode'] == 'fit':
        if kwargs['ckpt_path']:
            pl_trainer.fit(
                model = pl_model,
                datamodule = dm)
                # ckpt_path = kwargs['ckpt_path'])
        else:
            pl_trainer.fit(
                model = pl_model,
                datamodule = dm)

    if kwargs['mode'] == 'test':
        if kwargs['ckpt_path']:
            pl_trainer.test(
                model = pl_model,
                datamodule = dm,
                ckpt_path = kwargs['ckpt_path'])
        else:
            print("No ckpt_path,CAN NOT USE UNTRAINED MODEL FOR TEST")
            return False
    
    del dm, pl_model, pl_trainer
    torch.cuda.empty_cache()
    return True


def generate_data_bc(f, a, path='./data/', minQ=1, maxQ=2, n=129, train_N=2000, val_N=10, ax=0):
    p = Path(path)
    if not p.is_dir():
        p.mkdir(exist_ok=True)
    h = (2*a)/(n-1)

    x = np.linspace(-a, a, n)
    y = np.linspace(-a, a, n)
    xx, yy = np.meshgrid(x, y)

    mask = np.ones_like(xx)
    mask[1:-1, 1:-1] *= 0 

    train_Qs = np.linspace(minQ, maxQ, train_N)
    val_Qs = np.linspace(minQ, maxQ, val_N)
    
    if 'normal' in f.__name__:
        f1_mat = f(xx, yy, h)
    else:
        f1_mat = f(xx, yy)

    F = np.array(list(np.stack([xx, yy, q * f1_mat], axis=ax) for q in train_Qs))
    np.save(path+'F.npy', F)
    del F
    
    ValF = np.array(list(np.stack([xx, yy, q * f1_mat], axis=ax) for q in val_Qs))
    np.save(path+'ValF.npy', ValF)
    del ValF

    M = np.array(list(np.stack([mask, q * f1_mat], axis=ax) for q in train_Qs))
    np.save(path+'M.npy', M)
    del M
    
    ValM = np.array(list(np.stack([mask, q * f1_mat], axis=ax) for q in val_Qs))
    np.save(path+'ValM.npy', ValM)
    del ValM

    A2nd_bc = fd_A_with_bc(n, 2)
    D2nd_bc = ((A2nd_bc - sparse.diags(A2nd_bc.diagonal())) / 4).tocoo()
    sparse.save_npz(path+'A2nd', A2nd_bc)
    sparse.save_npz(path+'D2nd', D2nd_bc)
    del A2nd_bc, D2nd_bc

    A4th_bc = fd_A_with_bc(n, 4)
    D4th_bc = ((A4th_bc - sparse.diags(A4th_bc.diagonal())) / 20).tocoo()
    sparse.save_npz(path+'A4th', A4th_bc)
    sparse.save_npz(path+'D4th', D4th_bc)
    del A4th_bc, D4th_bc

    b2nd_bc = fd_b_bc(f1_mat, h, 2)
    B2nd_bc = np.array(list(b2nd_bc * q for q in train_Qs))
    np.save(path+'B2nd.npy', np.array(B2nd_bc))
    del B2nd_bc

    b4th_bc = fd_b_bc(f1_mat, h, 4)
    B4th_bc = np.array(list(b4th_bc * q for q in train_Qs))
    np.save(path+'B4th.npy', np.array(B4th_bc))
    del B4th_bc

    valB2nd_bc = np.array(list(b2nd_bc * q for q in val_Qs))
    np.save(path+'ValB2nd.npy', np.array(valB2nd_bc))
    del valB2nd_bc

    valB4th_bc = np.array(list(b4th_bc * q for q in val_Qs))
    np.save(path+'ValB4th.npy', np.array(valB4th_bc))
    del valB4th_bc

    return True

def generate_data(f, a, bc=False, path='./data/', minQ=1, maxQ=2, n=130, train_N=500, val_N=5, ax=0):

    p = Path(path)
    if not p.is_dir():
        p.mkdir(exist_ok=True)

    h = (2*a)/(n-1)
    x = np.linspace(-a, a, n)
    y = np.linspace(-a, a, n)
    xx, yy = np.meshgrid(x, y)
    
    mask = np.ones_like(xx)
    mask[1:-1, 1:-1] *= 0

    train_Qs = np.linspace(minQ, maxQ, train_N)
    val_Qs = np.linspace(minQ, maxQ, val_N)
    
    if 'normal' in f.__name__:
        f1_mat = f(xx, yy, h)
    else:
        f1_mat = f(xx, yy)

    F = np.array(list(np.stack([xx, yy, q * f1_mat], axis=ax) for q in train_Qs))
    np.save(path+'F.npy', F)
    del F

    ValF = np.array(list(np.stack([xx, yy, q * f1_mat], axis=ax) for q in val_Qs))
    np.save(path+'ValF.npy', ValF)
    del ValF

    M = np.array(list(np.stack([mask, q * f1_mat], axis=ax) for q in train_Qs))
    np.save(path+'M.npy', M)
    del M
    
    ValM = np.array(list(np.stack([mask, q * f1_mat], axis=ax) for q in val_Qs))
    np.save(path+'ValM.npy', ValM)
    del ValM

    A2nd = fd_A(n, 2)
    D2nd = ((A2nd - sparse.diags(A2nd.diagonal())) / 4).tocoo()
    sparse.save_npz(path+'A2nd', A2nd)
    sparse.save_npz(path+'D2nd', D2nd)
    del A2nd, D2nd

    A4th = fd_A(n, 4)
    D4th = ((A4th - sparse.diags(A4th.diagonal())) / 20).tocoo()
    sparse.save_npz(path+'A4th', A4th)
    sparse.save_npz(path+'D4th', D4th)
    del A4th, D4th

    b2nd = fd_b(f1_mat, h, 2)
    B2nd = np.array(list(b2nd * q for q in train_Qs))
    np.save(path+'B2nd.npy', np.array(B2nd))
    del B2nd

    b4th = fd_b(f1_mat, h, 4)
    B4th = np.array(list(b4th * q for q in train_Qs))
    np.save(path+'B4th.npy', np.array(B4th))
    del B4th

    valB2nd = np.array(list(b2nd * q for q in val_Qs))
    np.save(path+'ValB2nd.npy', np.array(valB2nd))
    del valB2nd

    valB4th = np.array(list(b4th * q for q in val_Qs))
    np.save(path+'ValB4th.npy', np.array(valB4th))
    del valB4th

    return True

def gen_test_data(Qs, n, f, a=1, order=2, g=0, path='./data/test/'):
    p = Path(path)
    if not p.is_dir():  p.mkdir(exist_ok=False)
    
    h = (2*a)/(n-1)
    x, y = np.linspace(-a, a, n), np.linspace(-a, a, n)
    xx, yy = np.meshgrid(x, y)
    f_mat = f(xx, yy, h)
    
    A = fd_A_with_bc(n, order).tocsr()
    for q in Qs:
        b = fd_b_bc(f_mat, h, order)
        b = apply_dirichlet_bc(b, True, g, order)
        u = spsolve(A, b)
        np.save(f'{path}{q:.3f}.npy', u)

    return True

# if __name__ == '__main__':
#     yitas = [yita11_2d, yita12_2d, yita22_2d, yita23_2d, yita25_2d, yita2cos_2d]

#     generate_data(f=normal, a=1, path='./data/normal257/', minQ=1, maxQ=2, n=256, train_N=2000, val_N=10)
#     generate_data(f=normal, a=1, path='./data/normal129/', minQ=1, maxQ=2, n=129, train_N=2000, val_N=10)
#     generate_data(f=normal, a=1, path='./data/normal65/',  minQ=1, maxQ=2, n=65, train_N=2000, val_N=10)
#     generate_data(f=normal, a=1, path='./data/normal33/', minQ=1, maxQ=2, n=33, train_N=2000, val_N=10)

#     generate_data(f=normal_fourth, a=1, path='./data/4normal257/', minQ=1, maxQ=2, n=256, train_N=2000, val_N=10)
#     generate_data(f=normal_fourth, a=1, path='./data/4normal129/', minQ=1, maxQ=2, n=129, train_N=2000, val_N=10)
#     generate_data(f=normal_fourth, a=1, path='./data/4normal65/',  minQ=1, maxQ=2, n=65, train_N=2000, val_N=10)
#     generate_data(f=normal_fourth, a=1, path='./data/4normal33/', minQ=1, maxQ=2, n=33, train_N=2000, val_N=10)

#     generate_data_bc(f=normal, a=1, path='./data/normal257_bc/', minQ=1, maxQ=2, n=256, train_N=2000, val_N=10)
#     generate_data_bc(f=normal, a=1, path='./data/normal129_bc/', minQ=1, maxQ=2, n=129, train_N=2000, val_N=10)
#     generate_data_bc(f=normal, a=1, path='./data/normal65_bc/',  minQ=1, maxQ=2, n=65, train_N=2000, val_N=10)
#     generate_data_bc(f=normal, a=1, path='./data/normal33_bc/', minQ=1, maxQ=2, n=33, train_N=2000, val_N=10)

#     generate_data_bc(f=normal_fourth, a=1, path='./data/4normal257_bc/', minQ=1, maxQ=2, n=256, train_N=2000, val_N=10)
#     generate_data_bc(f=normal_fourth, a=1, path='./data/4normal129_bc/', minQ=1, maxQ=2, n=129, train_N=2000, val_N=10)
#     generate_data_bc(f=normal_fourth, a=1, path='./data/4normal65_bc/',  minQ=1, maxQ=2, n=65, train_N=2000, val_N=10)
#     generate_data_bc(f=normal_fourth, a=1, path='./data/4normal33_bc/', minQ=1, maxQ=2, n=33, train_N=2000, val_N=10)

if __name__ == '__main__':
    Qs = np.linspace(0.5, 2.5, 10)
    gen_test_data(Qs, 1025, normal)
