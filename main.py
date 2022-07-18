
from data_module import DataModule, LinalgDataModule, SinDataModule
import pytorch_lightning as pl
from models import *
from pl_trainer import *
from utils import *
from pytorch_lightning.callbacks import ModelCheckpoint

case1_params = {
    # "pl_dataModule":DataModule(path = './data/sin/',
    #                            batch_size = 8),

    "pl_dataModule":SinDataModule(h=0.01, batch_size=8),
    "check_point": ModelCheckpoint(**{'monitor': 'Val L1 Loss', 
                                      'mode': 'min', 
                                      'every_n_train_steps': 0, 
                                      'every_n_epochs': 1, 
                                      'train_time_interval': None, 
                                      'save_on_train_epoch_end': True}),

    "pl_model": Hard_FD_Rec_Module( #net = AttUNet(in_c=3, out_c=1),
                                    net = UNet(1, 3, 2),
                                    loss = WeightedLoss(
                                            diff_fun = F.l1_loss,
                                            alpha = 10.0,
                                            beta = 0.0),
                                    fig_save_path = './u/sin/',
                                    lr = 1e-3,
                                    rhs = Dir_RHS_Rec(
                                        k=1, h=0.01, value=0
                                    )),
    "gpus": 1,
    "max_epochs": 80,
    "precision": 32,
    "check_val_every_n_epoch":1,
    # "ckpt_path": "./lightning_logs/version_0/checkpoints/epoch=36-step=4587.ckpt",
    "ckpt_path": False,
    "mode":"fit"
}

case2_params = {
    "pl_dataModule":LinalgDataModule(train_b='./B2nd.npy', train_u='./U2nd.npy', train_f='./F2nd.npy',
                                     val_b='./ValB2nd.npy', val_u='./ValU2nd.npy', val_f='./ValF2nd.npy',
                                    batch_size=2, N=128),
    "check_point": ModelCheckpoint(**{'monitor': 'Val linalg Loss', 
                                      'mode': 'min', 
                                      'every_n_train_steps': 0, 
                                      'every_n_epochs': 1, 
                                      'train_time_interval': None, 
                                      'save_on_train_epoch_end': True}),

    "pl_model": LinalgTrainer(net = UNet(1, 3, 2),
                              loss = F.mse_loss,
                              val_save_path='./u/lin1e-2/',
                              A = './A2nd.npz',
                              lr=1e-3),
    "gpus": 1,
    "max_epochs": 50,
    "precision": 32,
    "check_val_every_n_epoch":1,
    # "ckpt_path": "./lightning_logs/version_0/checkpoints/epoch=65-step=16368.ckpt",
    "ckpt_path": False,
    "mode":"fit"
}

case3_params = {
    "pl_dataModule":LinalgDataModule(train_b='./B4th.npy', train_u='./U4th.npy', train_f='./F4th.npy',
                                     val_b='./ValB4th.npy', val_u='./ValU4th.npy', val_f='./ValF4th.npy',
                                    batch_size=2, N=128),
    "check_point": ModelCheckpoint(**{'monitor': 'Val linalg Loss', 
                                      'mode': 'min', 
                                      'every_n_train_steps': 0, 
                                      'every_n_epochs': 1, 
                                      'train_time_interval': None, 
                                      'save_on_train_epoch_end': True}),

    "pl_model": LinalgTrainer(net = UNet(1, 3, 2),
                              loss = F.mse_loss,
                              val_save_path='./u/lin1e-2/',
                              A = './A4th.npz',
                              lr=1e-3),
    "gpus": 1,
    "max_epochs": 70,
    "precision": 32,
    "check_val_every_n_epoch":1,
    # "ckpt_path": "./lightning_logs/version_0/checkpoints/epoch=65-step=16368.ckpt",
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
    main(case3_params)