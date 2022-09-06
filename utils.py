import torch
import pytorch_lightning as pl

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
