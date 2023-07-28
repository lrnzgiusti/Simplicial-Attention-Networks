#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains utility functions to handle various mathematical 
operations commonly used in deep learning with simplicial complexes and 
non-Euclidean spaces. 

Functions include matrix normalization, sparse matrix conversions, and 
batch matrix multiplications.

@author: Lorenzo Giusti
"""

import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import numpy as np



def compute_projection_matrix(L, eps, kappa):
    """
    Approximate the limit of a power series to compute a projection matrix.
    
    Parameters
    ----------
    L : torch.Tensor
        The input tensor to be processed.
    eps : float
        A small number used to ensure numerical stability.
    kappa : int
        The number of times to multiply the projection matrix by itself.
    
    Returns
    -------
    P : torch.Tensor
        The approximated projection matrix.
    """
    P = (torch.eye(L.shape[0]) - eps*L)
    for _ in range(kappa):
        P = P @ P  # approximate the limit
    return P



def normalize(L, half_interval=False):
    """
    Normalize a tensor using the largest eigenvalue.

    Parameters
    ----------
    L : torch.Tensor
        The tensor to be normalized.
    half_interval : bool, optional
        Determines the interval used for normalization. Defaults to False.

    Returns
    -------
    torch.Tensor
        The normalized tensor.
    """
    assert(L.shape[0] == L.shape[1])
    topeig = torch.lobpcg(L, largest=True)[0]
    values = L.values()
    if half_interval:
        values *= 1.0/topeig
    else:
        values *= 2.0/topeig

    return torch.sparse_coo_tensor(L.indices(), values,   size=L.shape).to_dense()


def coo2tensor(A):
    """
    Convert a sparse matrix in COO format to a tensor.

    Parameters
    ----------
    A : scipy.sparse.coo.coo_matrix
        The sparse matrix in COO format.

    Returns
    -------
    torch.sparse_coo_tensor
        The corresponding tensor.
    """
    assert(sp.isspmatrix_coo(A))
    idxs = torch.LongTensor(np.vstack((A.row, A.col)))
    vals = torch.FloatTensor(A.data)
    return torch.sparse_coo_tensor(idxs, vals, size = A.shape, requires_grad = False)


def normalize2(L,Lx, half_interval = False):
    """
    Normalizes a sparse matrix using the largest eigenvalue.

    Parameters
    ----------
    L : scipy.sparse.csr.csr_matrix
        The sparse matrix to be normalized.
    Lx : scipy.sparse.csr.csr_matrix
        A copy of the sparse matrix `L`.
    half_interval : bool, optional
        Determines the interval used for normalization. Defaults to False.

    Returns
    -------
    scipy.sparse.csr.csr_matrix
        The normalized sparse matrix.
    """
    assert(sp.isspmatrix(L))
    M = L.shape[0]
    assert(M == L.shape[1])
    topeig = spl.eigsh(L, k=1, which="LM", return_eigenvectors = False)[0]    # we use the maximal eigenvalue of L to normalize
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
    Performs matrix multiplication for a batch of matrices.

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix to be multiplied, size (m, n).
    matrix_batch : torch.Tensor
        The batch of matrices to be multiplied, size (b, n, k).

    Returns
    -------
    torch.Tensor
        The batched matrix-matrix product, size (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)
    return matrix.mm(vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0)



def coo2tensor(A):
    assert(sp.isspmatrix_coo(A))
    idxs = torch.LongTensor(np.vstack((A.row, A.col)))
    vals = torch.FloatTensor(A.data)
    return torch.sparse_coo_tensor(idxs, vals, size = A.shape, requires_grad = False)
