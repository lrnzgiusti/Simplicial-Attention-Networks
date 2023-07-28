#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:00:00 2022

@author: Lorenzo Giusti
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SALayer(nn.Module):
    """
    SALayer is a simplicial attention layer for signals on simplicial complexes.
    It processes the attention mechanism on upper (solenoidal) and lower (irrotational) neighbourhoods,
    as well as the harmonic component of an edge signal, providing a generalization of graph attention
    networks to higher-order domains.
    """
    def __init__(self, F_in, F_out, L, kappa, p_dropout, alpha_leaky_relu):
        """
        Initialize the SALayer.
        Args:
            F_in (int): Number of features of the input signal.
            F_out (int): Number of features of the output component.
            L (tuple): Tuple containing matrices defining the simplicial complex structure.
            kappa (int): Maximum order of neighborhood to consider in attention computation.
            p_dropout (float): Dropout rate.
            alpha_leaky_relu (float): Negative slope coefficient for LeakyReLU.
        """
        super(SALayer, self).__init__()

        self.K = kappa
        self.F_in = F_in
        self.F_out = F_out
        self.L = L
        # Initialize the transformation matrices for the three types of components: 
        # Irr, Sol, and Har.
        self.Wirr = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)))
        self.Wsol = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)))
        self.Whar = nn.Parameter(torch.empty(size=(F_in, F_out)))
        # Initialize the attention matrices.
        self.att_irr = nn.Parameter(torch.empty(size=(2*F_out*self.K, 1)))
        self.att_sol = nn.Parameter(torch.empty(size=(2*F_out*self.K, 1)))

        self.dropout = p_dropout
        self.leakyrelu = nn.LeakyReLU(alpha_leaky_relu)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Wirr.data, gain=gain)
        nn.init.xavier_uniform_(self.Wsol.data, gain=gain)
        nn.init.xavier_uniform_(self.Whar.data, gain=gain)
        nn.init.xavier_uniform_(self.att_irr.data, gain=gain)
        nn.init.xavier_uniform_(self.att_sol.data, gain=gain)
        
    def forward(self, x):
        """
        Perform the forward computation of the layer.
        Args:
            x (torch.Tensor): Input signal with shape (num_nodes, F_in).
        Returns:
            torch.Tensor: Output signal with shape (num_nodes, F_out).
        """
        Ld, Lu , P = self.L

        # Compute the Irrotational, Solenoidal and Harmonic component transformations.
        # (ExE) x (ExF_in) x (F_inxF_out) -> (ExF_out)
        x_irr = torch.cat([x @ self.Wirr[k] for k in range(self.K)], dim=1)
        # (ExE) x (ExF_in) x (F_inxF_out) -> (ExF_out)
        x_sol = torch.cat([x @ self.Wsol[k] for k in range(self.K)], dim=1)
        # (ExE) x (ExF_in) x (F_inxF_out) -> (ExF_out)
        x_har = P @ x @ self.Whar

        # Compute attention coefficients. # Broadcast add: (Ex1) + (1xE) -> (ExE)
        
        E_irr = self.leakyrelu((x_irr @ self.att_irr[:self.F_out*self.K, :]) + (x_irr @ self.att_irr[self.F_out*self.K:, :]).T)
        E_sol = self.leakyrelu((x_sol @ self.att_sol[:self.F_out*self.K, :]) + (x_sol @ self.att_sol[self.F_out*self.K:, :]).T)
        
        # Exclude self-loops by setting diagonal entries to a large negative value.
        zero_vec = -9e15*torch.ones_like(E_irr)
        E_irr = torch.where(Ld != 0, E_irr, zero_vec)
        E_sol = torch.where(Lu != 0, E_sol, zero_vec)

        # Broadcast add
        alpha_irr = F.dropout(F.softmax(
            E_irr, dim=1), self.dropout, training=self.training) # (ExE) -> (ExE)
        alpha_sol = F.dropout(F.softmax(
            E_sol, dim=1), self.dropout, training=self.training) # (ExE) -> (ExE)


        # Apply softmax to compute the final attention coefficients.
        alpha_irr = F.dropout(F.softmax(E_irr, dim=1), self.dropout, training=self.training)
        alpha_sol = F.dropout(F.softmax(E_sol, dim=1), self.dropout, training=self.training)

        # Compute the final output signal.
        alpha_irr_k =  torch.clone(alpha_irr)
        alpha_sol_k = torch.clone(alpha_sol)
        z_i = alpha_irr_k @ torch.clone(x  @ self.Wirr[0])
        z_s = alpha_sol_k @ torch.clone(x  @ self.Wsol[0])
        for k in range(1, self.K):
            alpha_irr_k = alpha_irr_k @ Ld 
            alpha_sol_k = alpha_sol_k @ Lu 
            z_i += alpha_irr_k  @  x  @ self.Wirr[k]
            z_s += alpha_sol_k  @  x  @ self.Wsol[k]

        out = (z_i + z_s + x_har)
        return out


class SCLayer(nn.Module):

    def __init__(self, F_in, F_out, L, kappa, p_dropout, alpha_leaky_relu):
        """
        F_in: Numer of features of the input signal
        F_out: Numer of features of the output *component*
        """
        super(SCLayer, self).__init__()
        self.K = kappa
        self.F_in = F_in
        self.F_out = F_out
        self.Wirr = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)))
        self.Wsol = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)))
        self.Whar = nn.Parameter(torch.empty(size=(F_in, F_out)))


        self.dropout = p_dropout  # 0.0#0.6
        self.leakyrelu = nn.LeakyReLU(alpha_leaky_relu)

        self.L = L
        self.reset_parameters()

        print("Created SCLayer")

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Wirr.data, gain=gain)
        nn.init.xavier_uniform_(self.Wsol.data, gain=gain)
        nn.init.xavier_uniform_(self.Whar.data, gain=gain)

    def forward(self, x):
        Ld, Lu , P = self.L


        x_har = P @ x @ self.Whar
        
        alpha_irr_k = torch.clone(Ld).requires_grad_(False)
        alpha_sol_k = torch.clone(Lu).requires_grad_(False)

        z_i = alpha_irr_k @ torch.clone(x  @ self.Wirr[0])
        z_s = alpha_sol_k @ torch.clone(x  @ self.Wsol[0])
        for k in range(1, self.K):
            alpha_irr_k = alpha_irr_k @ Ld # alpha_irr
            alpha_sol_k = alpha_sol_k @ Lu #alpha_sol
            z_i += alpha_irr_k  @  x  @ self.Wirr[k]
            # (ExE) x (ExF_out) -> (ExF_out)
            z_s += alpha_sol_k  @  x  @ self.Wsol[k]

        out = F.dropout((z_i + z_s + x_har), self.dropout,
                        training=self.training)  # (ExE) -> (ExE)
        return out
