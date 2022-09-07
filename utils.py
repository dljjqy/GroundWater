import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import MyDataModule
from models import *
from trainer import *
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

def gen_hyper_dict(gridSize, batch_size, order, mode, net, features, four=False,
                    label='jac', lr=1e-3, bc=False, max_epochs=80, ckpt=False, lbfgs=False):
    exp_name = f'{mode:s}_{gridSize}_{net}_{features}_bs{batch_size}_{label}_order{order}_lr{lr:.0e}'
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
    if lbfgs:
        exp_name = 'lbfgs_' + exp_name

    if ckpt:
        exp_name = 'resume_' + exp_name

    dc = {'max_epochs':max_epochs, 'precision':32, 'check_val_every_n_epoch':1, 'ckpt_path':ckpt, 'mode':'fit', 'gpus':1}
    dc['pl_dataModule'] = MyDataModule(data_path, batch_size, order, mode)
    dc['check_point'] = ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', mode = 'min', every_n_train_steps = 0, 
                            every_n_epochs = 1, train_time_interval = None, save_top_k = 3, save_last = True,)
    dc['logger'] = TensorBoardLogger('./lightning_logs/', exp_name)
    
    if  lbfgs:
        dc['pl_model'] = pl_lbfgs_Model(F.l1_loss, model, f'./valu/{exp_name}/', data_path, lr, order)
    else:
        dc['pl_model'] = pl_Model(F.l1_loss, model, f'./valu/{exp_name}/', data_path, lr, order)

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
                datamodule = dm,
                ckpt_path = kwargs['ckpt_path'])
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
