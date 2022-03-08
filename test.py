#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:43:56 2022

@author: ince
"""


import torch
import torch.nn as nn

import pytorch_lightning as pl

from architecture.simplicial_attention_network import SAN
from utils.data_util import *



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--max_epochs", help="Maximum number of epochs",
                    type=int, default=1000)
parser.add_argument("-att", "--attention", help="Abilitate attention mechanism",
                    type=str, default="T")
parser.add_argument("-f", "--features", help="number of per layer features i.e. [5,5]",
                    type=str, default="[5,5]")
parser.add_argument("-d", "--dense", help="number of per  dense layer features i.e. [5,5]",
                    type=str,  default="[]")
parser.add_argument("-lr", "--learning_rate", help="learning rate i.e. 0.001",
                    type=float, default=0.001)
parser.add_argument("-wd", "--weight_decay", help="l2 regularization term i.e. 0.001",
                    type=float, default=0.0)
parser.add_argument("-eps", "--eps_proj", help="epsilon value for computing projetion matrix i.e. 0.9",
                    type=float, default=0.0)
parser.add_argument("-Kp", "--k_proj", help="K value for computing projetion matrix i.e. 5",
                    type=int, default=0)
parser.add_argument("-k", "--kappa", help="kappa value for diffusion i.e. 5",
                    type=int, default=5)
parser.add_argument("-do", "--dropout", help="probability of dropout i.e. 0.6",
                    type=float, default=0.0)
parser.add_argument("-a", "--activation", help="activation function all lowercase",
                    type=str, default='leaky_relu')
parser.add_argument("-ns", "--negative_slope", help="negative slope leaky relu",
                    type=float, default=0.01)
parser.add_argument("-pm", "--pct_miss", help="pct of missing values (complex dataset)",
                    type=int, default=10)
parser.add_argument("-o", "--order", help="order of the simplex to load (complex dataset)",
                    type=int, default=0)
parser.add_argument("-en", "--exp_num", help="experimental setup to load (complex dataset)",
                    type=int, default=0)
parser.add_argument("-s", "--seed", help="random seed",
                    type=int, default=0)
parser.add_argument("-id", "--pci_id", help="id bus seed",
                    type=str, default="0")
args = parser.parse_args()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=args.pci_id

device = torch.device(
    "cuda" if torch.cuda.is_available() else torch.device("cpu"))

features = [int(f) for f in args.features[2:-2].split(",")]
try:
    dense = [int(f) for f in args.dense[2:-2].split(",")]
except:
    dense = []
lr = args.learning_rate
wd = args.weight_decay


activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'elu': nn.ELU(),
    'selu': nn.SELU(),
    'leaky_relu' : nn.LeakyReLU(args.negative_slope),
}

activation_function = activations[args.activation]



collab_cmplx = CollaborationComplex(pct_miss=args.pct_miss,
                                    order=args.order, 
                                    device=device,
                                    eps=args.eps_proj,
                                    kappa=args.kappa,
                                    num_exp=args.exp_num)

attention = True if args.attention == "T" else False
print("ATT: ", attention)

s = SAN(in_features=collab_cmplx.X.shape[1], 
        n_class=collab_cmplx.n, 
        L=collab_cmplx.L, 
        features=features,
        dense=dense,  
        lr=args.learning_rate,
        k_proj=args.k_proj,
        sigma=activation_function,
        kappa=args.kappa, 
        p_dropout=args.dropout, 
        alpha_leaky_relu=args.negative_slope,
        attention=attention,
        device=device).to(device)




train_loader = \
    torch.utils.data.DataLoader(
        collab_cmplx, batch_size=None, batch_sampler=None, shuffle=True, num_workers=0)

string = "Test_citation"
logger = pl.loggers.TensorBoardLogger(name=string, save_dir='results')


pl.seed_everything(args.seed)


trainer = pl.Trainer(max_epochs=args.max_epochs, logger=logger,
                     gpus=1, auto_select_gpus=False)

trainer.fit(s, train_loader)


print("\n\n######")
print("######")
print("Max Accuracy:", s.max_acc.item())
print("######")
print("######\n\n")
