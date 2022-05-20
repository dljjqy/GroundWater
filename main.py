
from data_module import DataModule
import pytorch_lightning as pl
from models import *
from pl_trainer import *
from utils import *
from pytorch_lightning.callbacks import ModelCheckpoint

case1_params = {
    "pl_dataModule":DataModule(path = './data/case1/',
                               batch_size = 2),
                               
    "check_point": ModelCheckpoint(**{'monitor': 'Val L1 Loss', 
                                      'mode': 'min', 
                                      'every_n_train_steps': 0, 
                                      'every_n_epochs': 1, 
                                      'train_time_interval': None, 
                                      'save_on_train_epoch_end': True}),

    "pl_model": Hard_FD_Rec_Module( net = UNet(out_channels = 1,
                                               in_channels = 3,
                                               factor = 1),
                                    loss = WeightedLoss(
                                            diff_fun = F.l1_loss,
                                            alpha = 10.0,
                                            beta = 0.0),
                                    fig_save_path = './u/case1/',
                                    lr = 1e-3,
                                    rhs = Dir_RHS_Rec(
                                        k=1, h=0.005, value=0
                                    )),
    "gpus": 1,
    "max_epochs": 50,
    "precision": 32,
    "check_val_every_n_epoch":1,
    # "ckpt_path": "./lightning_logs/version_0/checkpoints/epoch=36-step=4587.ckpt",
    "ckpt_path": False,
    "mode":"fit"
}

case2_params = {
    "pl_dataModule":DataModule(path = './data/case2/',
                               batch_size = 2),
                               
    "check_point": ModelCheckpoint(**{'monitor': 'Val L1 Loss', 
                                      'mode': 'min', 
                                      'every_n_train_steps': 0, 
                                      'every_n_epochs': 1, 
                                      'train_time_interval': None, 
                                      'save_on_train_epoch_end': True}),

    "pl_model": Hard_FD_Rec_Module( net = UNet(out_channels = 1,
                                               in_channels = 3,
                                               factor = 1),
                                    loss = WeightedLoss(
                                            diff_fun = F.l1_loss,
                                            alpha = 10.0,
                                            beta = 0.0),
                                    fig_save_path = './u/case1/',
                                    lr = 1e-3,
                                    rhs = Dir_RHS_Rec(
                                        k=100, h=2.5, value=100
                                    )),
    "gpus": 1,
    "max_epochs": 50,
    "precision": 32,
    "check_val_every_n_epoch":1,
    # "ckpt_path": "./lightning_logs/version_0/checkpoints/epoch=36-step=4587.ckpt",
    "ckpt_path": False,
    "mode":"fit"
}

case3_params = {
    "pl_dataModule":DataModule(path = './data/case1/',
                               batch_size = 2),
                               
    "check_point": ModelCheckpoint(**{'monitor': 'Val L1 Loss', 
                                      'mode': 'min', 
                                      'every_n_train_steps': 0, 
                                      'every_n_epochs': 1, 
                                      'train_time_interval': None, 
                                      'save_on_train_epoch_end': True}),

    "pl_model": Hard_FD_Rec_Module( net = UNet(out_channels = 1,
                                               in_channels = 3,
                                               factor = 1),
                                    loss = WeightedLoss(
                                            diff_fun = F.l1_loss,
                                            alpha = 10.0,
                                            beta = 0.0),
                                    fig_save_path = './u/case1/',
                                    lr = 1e-3,
                                    rhs = XNeuYDir_RHS_Rec(
                                        k=1, h=0.005, dv=0, nv=0
                                    )),
    "gpus": 1,
    "max_epochs": 50,
    "precision": 32,
    "check_val_every_n_epoch":1,
    # "ckpt_path": "./lightning_logs/version_0/checkpoints/epoch=36-step=4587.ckpt",
    "ckpt_path": False,
    "mode":"fit"
}

case4_params = {
    "pl_dataModule":DataModule(path = './data/water/',
                               batch_size = 2),
                               
    "check_point": ModelCheckpoint(**{'monitor': 'Val L1 Loss', 
                                      'mode': 'min', 
                                      'every_n_train_steps': 0, 
                                      'every_n_epochs': 1, 
                                      'train_time_interval': None, 
                                      'save_on_train_epoch_end': True}),

    "pl_model": Hard_FD_Rec_Module( net = UNet(out_channels = 1,
                                               in_channels = 3,
                                               factor = 1),
                                    loss = WeightedLoss(
                                            diff_fun = F.l1_loss,
                                            alpha = 10.0,
                                            beta = 0.0),
                                    fig_save_path = './u/case1/',
                                    lr = 1e-3,
                                    rhs = XNeuYDir_RHS_Rec(
                                        k=100, h=2.5, dv=100, nv=0
                                    )),
    "gpus": 1,
    "max_epochs": 50,
    "precision": 32,
    "check_val_every_n_epoch":1,
    # "ckpt_path": "./lightning_logs/version_0/checkpoints/epoch=36-step=4587.ckpt",
    "ckpt_path": False,
    "mode":"fit"
}

# No Current method case1 green with Dirichlet boundary
case5_params = {
    "pl_dataModule":DataModule(path = './data/green/',
                               batch_size = 2),
                               
    "check_point": ModelCheckpoint(**{'monitor': 'Val L1 Loss', 
                                      'mode': 'min', 
                                      'every_n_train_steps': 0, 
                                      'every_n_epochs': 1, 
                                      'train_time_interval': None, 
                                      'save_on_train_epoch_end': True}),

    "pl_model": FiniteDiffModuleSoft( net = UNet(out_channels = 1,
                                               in_channels = 3,
                                               factor = 1),
                                    loss = WeightedLoss(
                                            diff_fun = F.l1_loss,
                                            alpha = 10.0,
                                            beta = 0.0),
                                    fig_save_path = './u/case1/',
                                    lr = 1e-3,
                                    rhs = Dir_RHS(
                                        k=1, h=0.005, value=0
                                    )),
    "gpus": 1,
    "max_epochs": 50,
    "precision": 32,
    "check_val_every_n_epoch":1,
    # "ckpt_path": "./lightning_logs/version_0/checkpoints/epoch=36-step=4587.ckpt",
    "ckpt_path": False,
    "mode":"fit"
}

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
        check_val_every_n_epoch = kwargs['check_val_every_n_epoch']
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
    return True

if __name__ == '__main__':
    main(case5_params)