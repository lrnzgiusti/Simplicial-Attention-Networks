#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:09:36 2022

@author: ince
"""


import pytorch_lightning as pl

from architecture.simplicial_attention_network import SAN
from utils.data_util import *




device = torch.device(
    "cuda" if torch.cuda.is_available() else torch.device("cpu"))


features = [5,5]
dense = []

collab_cmplx = CollaborationComplex(10, 0, 0)

s = SAN(in_features=1, n_class=collab_cmplx.n, L=collab_cmplx.L, features=features,
        dense=dense, device=device).to(device)


train_loader = \
    torch.utils.data.DataLoader(
        collab_cmplx, batch_size=None, batch_sampler=None, shuffle=True, num_workers=0)

val_loader = torch.utils.data.DataLoader(
    collab_cmplx, batch_size=None, batch_sampler=None, shuffle=False,  num_workers=0)

string = "Test_citation"
logger = pl.loggers.TensorBoardLogger(name=string, save_dir='results')


pl.seed_everything(0)


trainer = pl.Trainer(max_epochs=1000, logger=logger,
                     gpus=0, auto_select_gpus=False)

trainer.fit(s, train_loader, val_loader)




# %%

if False:
    train_data = OCEANDataset(split="train")
    val_data = OCEANDataset(split="val")
    
    incidences = train_data.get_incidences()
    
    
    features = [4]
    dense = []
    T = SAN(in_features=1, n_class=2, B_vec=incidences,
            features=features,
            dense=dense, device=device).to(device)
    
    train_loader = \
        torch.utils.data.DataLoader(
            train_data, batch_size=None, batch_sampler=None, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=None, batch_sampler=None, shuffle=False,  num_workers=0)
    
    # EXPERIMENT_[F1, F2]_[K1, K2]_[activation]_[lr]_[weight_decay]_[dropout]_[Batch_size_train_valid]_[N_epochs]_[epsilon]
    string = "Test"
    logger = pl.loggers.TensorBoardLogger(name=string, save_dir='results')
    
    
    pl.seed_everything(0)
    
    
    trainer = pl.Trainer(max_epochs=50, logger=logger,
                         gpus=0, auto_select_gpus=False)
    trainer.fit(T, train_loader, val_loader)
    
    print(T.max_acc)
    