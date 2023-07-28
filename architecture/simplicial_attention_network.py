#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Lorenzo Giusti
"""

import sys
# Add paths to system path to import required modules
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
# Import sparse matrix multiplication function from PyTorch
spmm = torch.sparse.mm


class SAN(pl.LightningModule):
    """
    The SAN class represents a Simplicial Attention Network, 
    a type of neural network that uses simplices 
    to model relationships between nodes in the network.

    Attributes:
    -----------
    N_dense_layers : int
        Number of dense layers in the model.
    N_simplicial_layers : int
        Number of simplicial layers in the model.
    L : list
        The Laplacian matrices for the simplicial complex. Each matrix is a torch.Tensor.
    dense : list
        List of output dimensions for each dense layer.
    lr : float
        Learning rate for the optimizer.
    E : int
        Number of simplices in the k-th simplex
    san : torch.nn.Sequential
        The sequence of operations representing the Simplicial Attention Network layers.
    mlp : torch.nn.Sequential
        The sequence of operations representing the multi-layer perceptron layers.
    max_acc : float
        The maximum accuracy achieved so far during training.
    loss_fn : torch.nn.Module
        The loss function used during training.
    train_acc : torchmetrics.Accuracy
        The training accuracy metric.
    val_acc : torchmetrics.Accuracy
        The validation accuracy metric.
    """
    def __init__(self, in_features, n_class, L, features, 
                       dense, lr, k_proj, sigma, kappa, 
                       p_dropout, alpha_leaky_relu, device, attention=True):
        """
        Initialize the SAN.

        Parameters
        ----------
        in_features : int
            The number of input features.
        n_class : int
            The number of classes for the classification task.
        L : list of torch.Tensor
            The Laplacian matrices for the simplicial complex.
        features : list of int
            List of number of features for each simplicial layer.
        dense : list of int
            List of output dimensions for each dense layer.
        lr : float
            Learning rate for the optimizer.
        k_proj : int
            The parameter for the number of iterations for the projection operation.
        sigma : torch.nn.Module
            The activation function for the simplicial layers.
        kappa : float
            The scaling factor for the attention mechanism.
        p_dropout : float
            The dropout probability for the dropout layer in the MLP.
        alpha_leaky_relu : float
            The negative slope coefficient for the LeakyReLU activation in the MLP.
        device : torch.device
            The device (CPU or GPU) where the tensors will be allocated.
        attention : bool, optional
            If True, use attention mechanism in simplicial layers. Default is True.
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

        # Save dense and learning rate parameters
        self.dense = dense
        self.dense.append(n_class)
        self.lr = lr

        self.N_dense_layers = len(dense)
        
        # Compute number of simplices in the k-th simplex
        E = L[0].shape[0] 
        self.L = [l.to(device) for l in L]

        # Create simplicial layers
        ops = []
        dropout = nn.Dropout(p=0.0)
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
            # Create a simplicial layer with attention if attention is True, otherwise create a standard layer
            simplicial_attention_layer = SALayer(**hparams).to(device) if attention else SCLayer(**hparams).to(device)
            ops.extend([simplicial_attention_layer, sigma])
        ops = ops[:-1]  # Remove the last activation function

        # Create dense layers (MLP)
        mlp = []
        simplicial_to_dense = nn.Linear(
            features[-1]*E, dense[0]).to(device)
        mlp.extend([simplicial_to_dense])
        for l in range(1, self.N_dense_layers):
            mlp.extend([sigma,  dropout, nn.Linear(dense[l-1], dense[l])])

        # Combine the simplicial layers and the MLP into sequences
        self.san = nn.Sequential(*ops)
        if n_class != E:
            self.mlp = nn.Sequential(*mlp)

        # Initialize metrics and loss
        self.max_acc = 0.0
        self.loss_fn = nn.L1Loss(reduction='mean')
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, x):
        """Define the forward pass of the model."""
        return self.san(x).view(-1, 1).T

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step in a single batch.

        Parameters
        ----------
        batch : tuple
            The input batch. It should be of the form (x, y, mask) where 'x' is
            the input data, 'y' is the corresponding targets, and 'mask' is used to mask
            the outputs. If 'mask' is not provided, it defaults to the range of 'y' length.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            The computed loss for the current training step.
        """
        # Unpack batch
        if len(batch) == 3:
            x, y, mask = batch
        else:
            x, y = batch
            mask = range(len(y))

        # Forward pass
        y_hat = self(x).squeeze(0)

        # Compute loss
        loss = self.loss_fn(y_hat[mask], y[mask])

        # Compute accuracy
        self.acc = ((y.float() - y_hat).abs() <= (0.05*y).abs() ).sum() / len(y)
        self.max_acc = max(self.acc, self.max_acc)

        # Log metrics
        self.log('valid_acc', self.acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step in a single batch.

        Parameters
        ----------
        batch : tuple
            The input batch. It should be of the form (x, y, mask) where 'x' is
            the input data, 'y' is the corresponding targets, and 'mask' is used to mask
            the outputs. If 'mask' is not provided, it defaults to the range of 'y' length.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            The computed loss for the current validation step.
        """
        x, y = batch
        if len(batch) == 3:
            x, y, mask = batch
        else:
            x, y = batch
            mask = range(len(y))

        # Forward pass
        y_hat = self(x).squeeze(0)

        # Compute loss
        loss = self.loss_fn(y_hat[mask], y[mask])
        return loss


    def test_step(self, batch, batch_idx):
        """
        Performs a single test step in a single batch. This simply calls the validation_step
        method as the operations performed are identical in this case.

        Parameters
        ----------
        batch : tuple
            The input batch. It should be of the form (x, y, mask).
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            The computed loss for the current test step.
        """
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outs):
        pass

    def validation_epoch_end(self, outs):
        pass

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training.

        Returns
        -------
        dict
            A dictionary containing the optimizer, the learning rate scheduler, and the metric to monitor.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3, weight_decay=0.0)
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=0.77,
                                      patience=100,
                                      min_lr=7e-5,
                                      verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}
