#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:02:48 2022

@author: ince
"""

import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("layers")
sys.path.append("utils")

import torch
import torchmetrics
import torch.nn as nn

import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from layers.simplicial_attention_layer import SALayer, SCLayer
from utils.utils import *
spmm = torch.sparse.mm


class SAN(pl.LightningModule):

    def __init__(self, in_features, n_class, L, features, 
                       dense, lr, k_proj, sigma, kappa, 
                       p_dropout, alpha_leaky_relu, device, attention=True):
        """
        

        Parameters
        ----------
        in_features : TYPE
            DESCRIPTION.
        n_class : TYPE
            DESCRIPTION.
        L : TYPE
            DESCRIPTION.
        features : TYPE
            DESCRIPTION.
        dense : TYPE
            DESCRIPTION.
        eps_proj : TYPE
            DESCRIPTION.
        k_proj : TYPE
            DESCRIPTION.
        sigma : TYPE
            DESCRIPTION.
        kappa : TYPE
            DESCRIPTION.
        p_dropout : TYPE
            DESCRIPTION.
        alpha_leaky_relu : TYPE
            DESCRIPTION.
        device : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(SAN, self).__init__()

        

        self.dense = dense
        self.dense.append(n_class)

        self.N_dense_layers = len(dense)
        
        E = L[0].shape[0] #number of simplices in the k-th simplex
        self.lr = lr
        K = k_proj
        #self.B1 = B_vec[0]
        #self.B2 = B_vec[1]

        """
        _Lu = [spmm(B_vec[i+1], B_vec[i+1].transpose(1, 0)
                    ).coalesce().to_dense() for i in range(len(B_vec)-1)]
        _Ld = [spmm(B_vec[i].transpose(1, 0),  B_vec[i]).coalesce().to_dense()
               for i in range(len(B_vec)-1)]

        _L = [spmm(B_vec[i].transpose(1, 0), B_vec[i]).coalesce() +
              spmm(B_vec[i+1], B_vec[i+1].transpose(1, 0)).coalesce() for i in range(len(B_vec)-1)]
        _L.insert(0, spmm(B_vec[0], B_vec[0].transpose(1, 0)).coalesce())
        _L.append(spmm(B_vec[-1].transpose(1, 0), B_vec[-1]).coalesce())

        L1 = normalize(_L[1])  # otherwise the harmonic projection blows up

        P = (torch.eye(self.E) - eps*L1).to(device)  # projection matrix -> ExE
        for i in range(K):
            P = P @ P  # approximate the limit

        # binary neighborhood mask on solenoidal component
        _Lu[0][_Lu[0] != 0] = 1
        # binary neighborhood mask on irrotational component
        _Ld[0][_Ld[0] != 0] = 1
        L = (_Lu[0], _Ld[0], P)  # Lu & Ld should be normalized?
        """

        self.L = [l.to(device) for l in L] #load from data

        ops = []
        dropout = nn.Dropout(p=0.0)
        sigma = sigma#nn.LeakyReLU()

        in_features = [in_features] + [features[l]
                                       for l in range(len(features))]

        self.N_simplicial_layers = len(in_features)
        for l in range(self.N_simplicial_layers-1):
            hparams = {"F_in":in_features[l],
                       "F_out":in_features[l+1],
                       "L":self.L, 
                       "kappa":kappa, 
                       "p_dropout":p_dropout, 
                       "alpha_leaky_relu":alpha_leaky_relu}
            simplicial_attention_layer = SALayer(**hparams).to(device) if attention else SCLayer(**hparams).to(device)
            ops.extend([simplicial_attention_layer, sigma])
        ops = ops[:-1]
        mlp = []
        simplicial_to_dense = nn.Linear(
            features[-1]*E, dense[0]).to(device)
        mlp.extend([simplicial_to_dense])
        for l in range(1, self.N_dense_layers):
            mlp.extend([sigma,  dropout, nn.Linear(dense[l-1], dense[l])])

        self.san = nn.Sequential(*ops)
        if n_class != E:
            self.mlp = nn.Sequential(*mlp)
        self.max_acc = 0.0
        self.loss_fn = nn.L1Loss(reduction='mean')
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.san(x).view(-1, 1).T

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, mask = batch
        else:
            x, y = batch
            mask = range(len(y))
        #y = y.unsqueeze(0)
        y_hat = self(x).squeeze(0)

        loss = self.loss_fn(y_hat[mask], y[mask])
        #self.train_acc(y_hat, y)

        #self.val_acc(y_hat, y)
        
        self.acc = ((y.float() - y_hat).abs() <= (0.05*y).abs() ).sum() / len(y)
        self.max_acc = max(self.acc, self.max_acc)
        self.log('valid_acc', self.acc, on_step=False,
                 on_epoch=True, prog_bar=True)
        #self.log('train_acc', self.train_acc, on_step=True,
        #         on_epoch=True, prog_bar=True)
        self.log('train_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if len(batch) == 3:
            x, y, mask = batch
        else:
            x, y = batch
            mask = range(len(y))
        y = y#.unsqueeze(0)
        y_hat = self(x).squeeze(0)

        loss = self.loss_fn(y_hat[mask], y[mask])
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outs):
        pass

    def validation_epoch_end(self, outs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3, weight_decay=0.0)
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=0.77,
                                      patience=100,
                                      min_lr=7e-5,
                                      verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}
