import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import MyDataModule
from models import UNet
from trainer import Trainer, TrainerLbfgs
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import main

data_mode = ['M1.npy', 'F1.npy', 'F2.npy']
val_data_mode = ['ValM1.npy', 'ValF1.npy', 'ValF2.npy']
# unet_67_16 = [
# {
#     "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=8, order=2, mode=0),
#     "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
#                                     mode = 'min', 
#                                     every_n_train_steps = 0, 
#                                     every_n_epochs = 1,
#                                     train_time_interval = None,
#                                     save_top_k = 3,
#                                     save_last = True,),
#     "logger" :TensorBoardLogger('./lightning_logs/', 'MUnet32_bs8_67_jac_order2_lr1e-3'),
#     "pl_model": Trainer(net = UNet(2, 1, 32, 1),
#                               loss = F.l1_loss,
#                               val_save_path='./valu/MUnet32_bs8_67_jac_order2_lr1e-3/',
#                               data_path='./data/normal67/',
#                               lr=1e-3, order=2),
#     "gpus": 1,
#     "max_epochs": 120,
#     "precision": 32,
#     "check_val_every_n_epoch":1,
#     "ckpt_path": False,
#     "mode":"fit"},
# {
#     "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=16, order=2, mode=0),
#     "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
#                                     mode = 'min', 
#                                     every_n_train_steps = 0, 
#                                     every_n_epochs = 1,
#                                     train_time_interval = None,
#                                     save_top_k = 3,
#                                     save_last = True,),
#     "logger" :TensorBoardLogger('./lightning_logs/', 'MUnet32_bs16_67_jac_order2_lr1e-3'),
#     "pl_model": Trainer(net = UNet(2, 1, 32, 1),
#                               loss = F.l1_loss,
#                               val_save_path='./valu/MUnet32_bs16_67_jac_order2_lr1e-3/',
#                               data_path='./data/normal67/',
#                               lr=1e-3, order=2),
#     "gpus": 1,
#     "max_epochs": 120,
#     "precision": 32,
#     "check_val_every_n_epoch":1,
#     "ckpt_path": False,
#     "mode":"fit"},
# {
#     "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=24, order=2, mode=0),
#     "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
#                                     mode = 'min', 
#                                     every_n_train_steps = 0, 
#                                     every_n_epochs = 1,
#                                     train_time_interval = None,
#                                     save_top_k = 3,
#                                     save_last = True,),
#     "logger" :TensorBoardLogger('./lightning_logs/', 'MUnet32_bs24_67_jac_order2_lr1e-3'),
#     "pl_model": Trainer(net = UNet(2, 1, 32, 1),
#                               loss = F.l1_loss,
#                               val_save_path='./valu/MUnet32_bs24_67_jac_order2_lr1e-3/',
#                               data_path='./data/normal67/',
#                               lr=1e-3, order=2),
#     "gpus": 1,
#     "max_epochs": 120,
#     "precision": 32,
#     "check_val_every_n_epoch":1,
#     "ckpt_path": False,
#     "mode":"fit"},
# {
#     "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=32, order=2, mode=0),
#     "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
#                                     mode = 'min', 
#                                     every_n_train_steps = 0, 
#                                     every_n_epochs = 1,
#                                     train_time_interval = None,
#                                     save_top_k = 3,
#                                     save_last = True,),
#     "logger" :TensorBoardLogger('./lightning_logs/', 'MUnet32_bs32_67_jac_order2_lr1e-3'),
#     "pl_model": Trainer(net = UNet(2, 1, 32, 1),
#                               loss = F.l1_loss,
#                               val_save_path='./valu/MUnet32_bs32_67_jac_order2_lr1e-3/',
#                               data_path='./data/normal67/',
#                               lr=1e-3, order=2),
#     "gpus": 1,
#     "max_epochs": 120,
#     "precision": 32,
#     "check_val_every_n_epoch":1,
#     "ckpt_path": False,
#     "mode":"fit"},

# # ******************************************

#     {
#     "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=8, order=2, mode=1),
#     "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
#                                     mode = 'min', 
#                                     every_n_train_steps = 0, 
#                                     every_n_epochs = 1,
#                                     train_time_interval = None,
#                                     save_top_k = 3,
#                                     save_last = True,),
#     "logger" :TensorBoardLogger('./lightning_logs/', 'F1Unet32_bs8_67_jac_order2_lr1e-3'),
#     "pl_model": Trainer(net = UNet(3, 1, 32, 1),
#                               loss = F.l1_loss,
#                               val_save_path='./valu/F1Unet32_bs8_67_jac_order2_lr1e-3/',
#                               data_path='./data/normal67/',
#                               lr=1e-3, order=2),
#     "gpus": 1,
#     "max_epochs": 120,
#     "precision": 32,
#     "check_val_every_n_epoch":1,
#     "ckpt_path": False,
#     "mode":"fit"},
# {
#     "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=16, order=2, mode=1),
#     "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
#                                     mode = 'min', 
#                                     every_n_train_steps = 0, 
#                                     every_n_epochs = 1,
#                                     train_time_interval = None,
#                                     save_top_k = 3,
#                                     save_last = True,),
#     "logger" :TensorBoardLogger('./lightning_logs/', 'F1Unet32_bs16_67_jac_order2_lr1e-3'),
#     "pl_model": Trainer(net = UNet(3, 1, 32, 1),
#                               loss = F.l1_loss,
#                               val_save_path='./valu/F1Unet32_bs16_67_jac_order2_lr1e-3/',
#                               data_path='./data/normal67/',
#                               lr=1e-3, order=2),
#     "gpus": 1,
#     "max_epochs": 120,
#     "precision": 32,
#     "check_val_every_n_epoch":1,
#     "ckpt_path": False,
#     "mode":"fit"},
# {
#     "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=24, order=2, mode=1),
#     "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
#                                     mode = 'min', 
#                                     every_n_train_steps = 0, 
#                                     every_n_epochs = 1,
#                                     train_time_interval = None,
#                                     save_top_k = 3,
#                                     save_last = True,),
#     "logger" :TensorBoardLogger('./lightning_logs/', 'F1Unet32_bs24_67_jac_order2_lr1e-3'),
#     "pl_model": Trainer(net = UNet(3, 1, 32, 1),
#                               loss = F.l1_loss,
#                               val_save_path='./valu/F1Unet32_bs24_67_jac_order2_lr1e-3/',
#                               data_path='./data/normal67/',
#                               lr=1e-3, order=2),
#     "gpus": 1,
#     "max_epochs": 120,
#     "precision": 32,
#     "check_val_every_n_epoch":1,
#     "ckpt_path": False,
#     "mode":"fit"},
# {
#     "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=32, order=2, mode=1),
#     "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
#                                     mode = 'min', 
#                                     every_n_train_steps = 0, 
#                                     every_n_epochs = 1,
#                                     train_time_interval = None,
#                                     save_top_k = 3,
#                                     save_last = True,),
#     "logger" :TensorBoardLogger('./lightning_logs/', 'F1Unet32_bs32_67_jac_order2_lr1e-3'),
#     "pl_model": Trainer(net = UNet(3, 1, 32, 1),
#                               loss = F.l1_loss,
#                               val_save_path='./valu/F1Unet32_bs32_67_jac_order2_lr1e-3/',
#                               data_path='./data/normal67/',
#                               lr=1e-3, order=2),
#     "gpus": 1,
#     "max_epochs": 120,
#     "precision": 32,
#     "check_val_every_n_epoch":1,
#     "ckpt_path": False,
#     "mode":"fit"},

# # *******************************
#     {
#     "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=8, order=2, mode=2),
#     "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
#                                     mode = 'min', 
#                                     every_n_train_steps = 0, 
#                                     every_n_epochs = 1,
#                                     train_time_interval = None,
#                                     save_top_k = 3,
#                                     save_last = True,),
#     "logger" :TensorBoardLogger('./lightning_logs/', 'F2Unet32_bs8_67_jac_order2_lr1e-3'),
#     "pl_model": Trainer(net = UNet(3, 1, 32, 2),
#                               loss = F.l1_loss,
#                               val_save_path='./valu/F2Unet32_bs8_67_jac_order2_lr1e-3/',
#                               data_path='./data/normal67/',
#                               lr=1e-3, order=2),
#     "gpus": 1,
#     "max_epochs": 120,
#     "precision": 32,
#     "check_val_every_n_epoch":1,
#     "ckpt_path": False,
#     "mode":"fit"},
# {
#     "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=16, order=2, mode=2),
#     "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
#                                     mode = 'min', 
#                                     every_n_train_steps = 0, 
#                                     every_n_epochs = 1,
#                                     train_time_interval = None,
#                                     save_top_k = 3,
#                                     save_last = True,),
#     "logger" :TensorBoardLogger('./lightning_logs/', 'F2Unet32_bs16_67_jac_order2_lr1e-3'),
#     "pl_model": Trainer(net = UNet(3, 1, 32, 2),
#                               loss = F.l1_loss,
#                               val_save_path='./valu/F2Unet32_bs16_67_jac_order2_lr1e-3/',
#                               data_path='./data/normal67/',
#                               lr=1e-3, order=2),
#     "gpus": 1,
#     "max_epochs": 120,
#     "precision": 32,
#     "check_val_every_n_epoch":1,
#     "ckpt_path": False,
#     "mode":"fit"},
# {
#     "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=24, order=2, mode=2),
#     "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
#                                     mode = 'min', 
#                                     every_n_train_steps = 0, 
#                                     every_n_epochs = 1,
#                                     train_time_interval = None,
#                                     save_top_k = 3,
#                                     save_last = True,),
#     "logger" :TensorBoardLogger('./lightning_logs/', 'F2Unet32_bs24_67_jac_order2_lr1e-3'),
#     "pl_model": Trainer(net = UNet(3, 1, 32, 2),
#                               loss = F.l1_loss,
#                               val_save_path='./valu/F2Unet32_bs24_67_jac_order2_lr1e-3/',
#                               data_path='./data/normal67/',
#                               lr=1e-3, order=2),
#     "gpus": 1,
#     "max_epochs": 120,
#     "precision": 32,
#     "check_val_every_n_epoch":1,
#     "ckpt_path": False,
#     "mode":"fit"},
# {
#     "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=32, order=2, mode=2),
#     "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
#                                     mode = 'min', 
#                                     every_n_train_steps = 0, 
#                                     every_n_epochs = 1,
#                                     train_time_interval = None,
#                                     save_top_k = 3,
#                                     save_last = True,),
#     "logger" :TensorBoardLogger('./lightning_logs/', 'F2Unet32_bs32_67_jac_order2_lr1e-3'),
#     "pl_model": Trainer(net = UNet(3, 1, 32, 2),
#                               loss = F.l1_loss,
#                               val_save_path='./valu/F2Unet32_bs32_67_jac_order2_lr1e-3/',
#                               data_path='./data/normal67/',
#                               lr=1e-3, order=2),
#     "gpus": 1,
#     "max_epochs": 120,
#     "precision": 32,
#     "check_val_every_n_epoch":1,
#     "ckpt_path": False,
#     "mode":"fit"},
# ]
unet_67_32 = [
{
    "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=8, order=2, mode=0),
    "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
                                    mode = 'min', 
                                    every_n_train_steps = 0, 
                                    every_n_epochs = 1,
                                    train_time_interval = None,
                                    save_top_k = 3,
                                    save_last = True,),
    "logger" :TensorBoardLogger('./lightning_logs/', 'MUnet32_bs8_67_jac_order2_lr1e-3'),
    "pl_model": Trainer(net = UNet(2, 1, 32, 1),
                              loss = F.l1_loss,
                              val_save_path='./valu/MUnet32_bs8_67_jac_order2_lr1e-3/',
                              data_path='./data/normal67/',
                              lr=1e-3, order=2),
    "gpus": 1,
    "max_epochs": 120,
    "precision": 32,
    "check_val_every_n_epoch":1,
    "ckpt_path": False,
    "mode":"fit"},
{
    "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=16, order=2, mode=0),
    "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
                                    mode = 'min', 
                                    every_n_train_steps = 0, 
                                    every_n_epochs = 1,
                                    train_time_interval = None,
                                    save_top_k = 3,
                                    save_last = True,),
    "logger" :TensorBoardLogger('./lightning_logs/', 'MUnet32_bs16_67_jac_order2_lr1e-3'),
    "pl_model": Trainer(net = UNet(2, 1, 32, 1),
                              loss = F.l1_loss,
                              val_save_path='./valu/MUnet32_bs16_67_jac_order2_lr1e-3/',
                              data_path='./data/normal67/',
                              lr=1e-3, order=2),
    "gpus": 1,
    "max_epochs": 120,
    "precision": 32,
    "check_val_every_n_epoch":1,
    "ckpt_path": False,
    "mode":"fit"},
{
    "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=24, order=2, mode=0),
    "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
                                    mode = 'min', 
                                    every_n_train_steps = 0, 
                                    every_n_epochs = 1,
                                    train_time_interval = None,
                                    save_top_k = 3,
                                    save_last = True,),
    "logger" :TensorBoardLogger('./lightning_logs/', 'MUnet32_bs24_67_jac_order2_lr1e-3'),
    "pl_model": Trainer(net = UNet(2, 1, 32, 1),
                              loss = F.l1_loss,
                              val_save_path='./valu/MUnet32_bs24_67_jac_order2_lr1e-3/',
                              data_path='./data/normal67/',
                              lr=1e-3, order=2),
    "gpus": 1,
    "max_epochs": 120,
    "precision": 32,
    "check_val_every_n_epoch":1,
    "ckpt_path": False,
    "mode":"fit"},
{
    "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=32, order=2, mode=0),
    "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
                                    mode = 'min', 
                                    every_n_train_steps = 0, 
                                    every_n_epochs = 1,
                                    train_time_interval = None,
                                    save_top_k = 3,
                                    save_last = True,),
    "logger" :TensorBoardLogger('./lightning_logs/', 'MUnet32_bs32_67_jac_order2_lr1e-3'),
    "pl_model": Trainer(net = UNet(2, 1, 32, 1),
                              loss = F.l1_loss,
                              val_save_path='./valu/MUnet32_bs32_67_jac_order2_lr1e-3/',
                              data_path='./data/normal67/',
                              lr=1e-3, order=2),
    "gpus": 1,
    "max_epochs": 120,
    "precision": 32,
    "check_val_every_n_epoch":1,
    "ckpt_path": False,
    "mode":"fit"},

# ******************************************

    {
    "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=8, order=2, mode=1),
    "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
                                    mode = 'min', 
                                    every_n_train_steps = 0, 
                                    every_n_epochs = 1,
                                    train_time_interval = None,
                                    save_top_k = 3,
                                    save_last = True,),
    "logger" :TensorBoardLogger('./lightning_logs/', 'F1Unet32_bs8_67_jac_order2_lr1e-3'),
    "pl_model": Trainer(net = UNet(3, 1, 32, 1),
                              loss = F.l1_loss,
                              val_save_path='./valu/F1Unet32_bs8_67_jac_order2_lr1e-3/',
                              data_path='./data/normal67/',
                              lr=1e-3, order=2),
    "gpus": 1,
    "max_epochs": 120,
    "precision": 32,
    "check_val_every_n_epoch":1,
    "ckpt_path": False,
    "mode":"fit"},
{
    "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=16, order=2, mode=1),
    "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
                                    mode = 'min', 
                                    every_n_train_steps = 0, 
                                    every_n_epochs = 1,
                                    train_time_interval = None,
                                    save_top_k = 3,
                                    save_last = True,),
    "logger" :TensorBoardLogger('./lightning_logs/', 'F1Unet32_bs16_67_jac_order2_lr1e-3'),
    "pl_model": Trainer(net = UNet(3, 1, 32, 1),
                              loss = F.l1_loss,
                              val_save_path='./valu/F1Unet32_bs16_67_jac_order2_lr1e-3/',
                              data_path='./data/normal67/',
                              lr=1e-3, order=2),
    "gpus": 1,
    "max_epochs": 120,
    "precision": 32,
    "check_val_every_n_epoch":1,
    "ckpt_path": False,
    "mode":"fit"},
{
    "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=24, order=2, mode=1),
    "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
                                    mode = 'min', 
                                    every_n_train_steps = 0, 
                                    every_n_epochs = 1,
                                    train_time_interval = None,
                                    save_top_k = 3,
                                    save_last = True,),
    "logger" :TensorBoardLogger('./lightning_logs/', 'F1Unet32_bs24_67_jac_order2_lr1e-3'),
    "pl_model": Trainer(net = UNet(3, 1, 32, 1),
                              loss = F.l1_loss,
                              val_save_path='./valu/F1Unet32_bs24_67_jac_order2_lr1e-3/',
                              data_path='./data/normal67/',
                              lr=1e-3, order=2),
    "gpus": 1,
    "max_epochs": 120,
    "precision": 32,
    "check_val_every_n_epoch":1,
    "ckpt_path": False,
    "mode":"fit"},
{
    "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=32, order=2, mode=1),
    "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
                                    mode = 'min', 
                                    every_n_train_steps = 0, 
                                    every_n_epochs = 1,
                                    train_time_interval = None,
                                    save_top_k = 3,
                                    save_last = True,),
    "logger" :TensorBoardLogger('./lightning_logs/', 'F1Unet32_bs32_67_jac_order2_lr1e-3'),
    "pl_model": Trainer(net = UNet(3, 1, 32, 1),
                              loss = F.l1_loss,
                              val_save_path='./valu/F1Unet32_bs32_67_jac_order2_lr1e-3/',
                              data_path='./data/normal67/',
                              lr=1e-3, order=2),
    "gpus": 1,
    "max_epochs": 120,
    "precision": 32,
    "check_val_every_n_epoch":1,
    "ckpt_path": False,
    "mode":"fit"},

# *******************************
    {
    "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=8, order=2, mode=2),
    "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
                                    mode = 'min', 
                                    every_n_train_steps = 0, 
                                    every_n_epochs = 1,
                                    train_time_interval = None,
                                    save_top_k = 3,
                                    save_last = True,),
    "logger" :TensorBoardLogger('./lightning_logs/', 'F2Unet32_bs8_67_jac_order2_lr1e-3'),
    "pl_model": Trainer(net = UNet(3, 1, 32, 2),
                              loss = F.l1_loss,
                              val_save_path='./valu/F2Unet32_bs8_67_jac_order2_lr1e-3/',
                              data_path='./data/normal67/',
                              lr=1e-3, order=2),
    "gpus": 1,
    "max_epochs": 120,
    "precision": 32,
    "check_val_every_n_epoch":1,
    "ckpt_path": False,
    "mode":"fit"},
{
    "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=16, order=2, mode=2),
    "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
                                    mode = 'min', 
                                    every_n_train_steps = 0, 
                                    every_n_epochs = 1,
                                    train_time_interval = None,
                                    save_top_k = 3,
                                    save_last = True,),
    "logger" :TensorBoardLogger('./lightning_logs/', 'F2Unet32_bs16_67_jac_order2_lr1e-3'),
    "pl_model": Trainer(net = UNet(3, 1, 32, 2),
                              loss = F.l1_loss,
                              val_save_path='./valu/F2Unet32_bs16_67_jac_order2_lr1e-3/',
                              data_path='./data/normal67/',
                              lr=1e-3, order=2),
    "gpus": 1,
    "max_epochs": 120,
    "precision": 32,
    "check_val_every_n_epoch":1,
    "ckpt_path": False,
    "mode":"fit"},
{
    "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=24, order=2, mode=2),
    "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
                                    mode = 'min', 
                                    every_n_train_steps = 0, 
                                    every_n_epochs = 1,
                                    train_time_interval = None,
                                    save_top_k = 3,
                                    save_last = True,),
    "logger" :TensorBoardLogger('./lightning_logs/', 'F2Unet32_bs24_67_jac_order2_lr1e-3'),
    "pl_model": Trainer(net = UNet(3, 1, 32, 2),
                              loss = F.l1_loss,
                              val_save_path='./valu/F2Unet32_bs24_67_jac_order2_lr1e-3/',
                              data_path='./data/normal67/',
                              lr=1e-3, order=2),
    "gpus": 1,
    "max_epochs": 120,
    "precision": 32,
    "check_val_every_n_epoch":1,
    "ckpt_path": False,
    "mode":"fit"},
{
    "pl_dataModule":MyDataModule(data_path='./data/normal67/', batch_size=32, order=2, mode=2),
    "check_point": ModelCheckpoint(monitor = 'Val Jacobian Iteration l1 Loss', 
                                    mode = 'min', 
                                    every_n_train_steps = 0, 
                                    every_n_epochs = 1,
                                    train_time_interval = None,
                                    save_top_k = 3,
                                    save_last = True,),
    "logger" :TensorBoardLogger('./lightning_logs/', 'F2Unet32_bs32_67_jac_order2_lr1e-3'),
    "pl_model": Trainer(net = UNet(3, 1, 32, 2),
                              loss = F.l1_loss,
                              val_save_path='./valu/F2Unet32_bs32_67_jac_order2_lr1e-3/',
                              data_path='./data/normal67/',
                              lr=1e-3, order=2),
    "gpus": 1,
    "max_epochs": 120,
    "precision": 32,
    "check_val_every_n_epoch":1,
    "ckpt_path": False,
    "mode":"fit"},
]

if __name__ == '__main__':
    for idx,d in enumerate(unet_67_32):
        print(f"\n ---The {idx+1} experiments--- \n")
        main(d)