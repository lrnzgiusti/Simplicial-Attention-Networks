#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:06:24 2022

@author: ince
"""

import torch

import scipy.sparse as sp
import scipy.sparse.linalg as spl
import numpy as np


def compute_projection_matrix(L, eps, kappa):
    P = (torch.eye(L.shape[0]) - eps*L)
    for _ in range(kappa):
        P = P @ P  # approximate the limit
    return P

def normalize(L, half_interval=False):
    assert(L.shape[0] == L.shape[1])
    topeig = torch.lobpcg(L, largest=True)[0]
    values = L.values()
    if half_interval:
        values *= 1.0/topeig
    else:
        values *= 2.0/topeig

    return torch.sparse_coo_tensor(L.indices(), values,   size=L.shape).to_dense()


def coo2tensor(A):
    assert(sp.isspmatrix_coo(A))
    idxs = torch.LongTensor(np.vstack((A.row, A.col)))
    vals = torch.FloatTensor(A.data)
    return torch.sparse_coo_tensor(idxs, vals, size = A.shape, requires_grad = False)

def normalize2(L,Lx, half_interval = False):
    assert(sp.isspmatrix(L))
    M = L.shape[0]
    assert(M == L.shape[1])
    topeig = spl.eigsh(L, k=1, which="LM", return_eigenvectors = False)[0]    # we use the maximal eigenvalue of L to normalize
    #print("Topeig = %f" %(topeig))

    ret = Lx.copy()
    if half_interval:
        ret *= 1.0/topeig
    else:
        ret *= 2.0/topeig
        ret.setdiag(ret.diagonal(0) - np.ones(M), 0)
        
    return ret




def normalize3(L, half_interval = False):
    assert(sp.isspmatrix(L))
    M = L.shape[0]
    assert(M == L.shape[1])
    topeig = spl.eigsh(L, k=1, which="LM", return_eigenvectors = False)[0]   
    #print("Topeig = %f" %(topeig))

    ret = L.copy()
    if half_interval:
        ret *= 1.0/topeig
    else:
        ret *= 2.0/topeig
        ret.setdiag(ret.diagonal(0) - np.ones(M), 0)

    return ret



def batch_mm(matrix, matrix_batch):
    """
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)
    #
    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping.
    #(m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    return matrix.mm(vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0)



def coo2tensor(A):
    assert(sp.isspmatrix_coo(A))
    idxs = torch.LongTensor(np.vstack((A.row, A.col)))
    vals = torch.FloatTensor(A.data)
    return torch.sparse_coo_tensor(idxs, vals, size = A.shape, requires_grad = False)