#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 15:21:51 2021

@author: ince
"""



from .utils import normalize, compute_projection_matrix, normalize2, coo2tensor
import torch
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict




class SyntheticVectorField(torch.utils.data.Dataset):
    def __init__(self, data_path=r"data/vec_ad/synth_vec_ad.pkl", split="train", device="cpu"):
        self.data = pickle.load(open(data_path, "rb"))
        self.X = torch.from_numpy(self.data['X']).float()
        self.y = torch.from_numpy(self.data['y']).long()
        idx = int(len(self.X)*0.8)
        if split == "train":

            self.X = self.X[:idx].to(device)
            self.y = self.y[:idx].to(device)

        else:

            self.X = self.X[idx:].to(device)
            self.y = self.y[idx:].to(device)

        # stored in scipy.coo_matrix
        self.boundaries = [self.data['B1'], self.data['B2']]

        # TODO 3: edge signal renderlo multifeature
        # TODO 4: se hai un segnale sui nodi, trasformarlo in un segnale sugli edge tramite una trasformazione non lineare e propagarlo

    def get_incidences(self):

        B1 = self.boundaries[0].A
        B2 = self.boundaries[1].A

        B1_t = torch.tensor(B1).to_sparse().to(torch.float32).coalesce()
        B2_t = torch.tensor(B2).to_sparse().to(torch.float32).coalesce()

        return [B1_t, B2_t]

    def __getitem__(self, index):

        X, y = self.X[index], self.y[index]
        return X, y[0]

    def __len__(self):
        # Returns length
        return len(self.X)


class SyntheticSIG(torch.utils.data.Dataset):
    def __init__(self, data_path=r"data/synth/synth_X_Y.pkl", split="train", device="cpu"):
        self.data = pickle.load(open(data_path, "rb"))
        self.X = torch.from_numpy(self.data['X']).T.float()
        self.y = torch.from_numpy(self.data['Y']).T.long()
        idx = int(len(self.X)*0.8)
        if split == "train":

            self.X = self.X[:idx].to(device)
            self.y = self.y[:idx].to(device)

        else:

            self.X = self.X[idx:].to(device)
            self.y = self.y[idx:].to(device)

        # stored in scipy.coo_matrix
        self.boundaries = [self.data['B1'], self.data['B2']]

    def get_incidences(self):

        B1 = self.boundaries[0]
        B2 = self.boundaries[1]

        B1_t = torch.tensor(B1).to_sparse().to(torch.float32).coalesce()
        B2_t = torch.tensor(B2).to_sparse().to(torch.float32).coalesce()

        return [B1_t, B2_t]

    def __getitem__(self, index):

        X, y = self.X[index], self.y[index]
        return X, y[0]

    def __len__(self):
        # Returns length
        return len(self.X)


class RNADataset(torch.utils.data.Dataset):
    def __init__(self, data_path=r"data/rna/rna_data.pkl",
                 split="train", store_boundaries=True):
        assert split in ["train", "val"]
        self.data = pickle.load(open(data_path, "rb"))
        # self.data = self.data[subset]
        self.y = torch.from_numpy(
            pd.get_dummies(self.data['y']).to_numpy()
        ).long().argmax(dim=1).to(device)
        self.Xe = torch.from_numpy(pickle.load(
            open("data/rna/Xe.pkl", "rb"))).reshape(-1, 1).to(device)
        self.Xn = torch.from_numpy(self.data['Xn']).to(device)
        self.boundaries = self.data['B']  # stored in scipy.coo_matrix
        idxs = np.arange(0, len(self.y)).astype(np.int64)
        self.train_mask = np.random.choice(idxs,
                                           int(len(self.y)*0.8),
                                           replace=False)
        self.valid_mask = np.setdiff1d(idxs, self.train_mask)

        # self.train_mask = torch.from_numpy(self.train_mask)
        # self.valid_mask = torch.from_numpy(self.valid_mask)
        if split == "train":
            self.mask = torch.from_numpy(self.train_mask)
        else:
            self.mask = torch.from_numpy(self.valid_mask)

          # TODO 3: edge signal renderlo multifeature
          # TODO 4: se hai un segnale sui nodi, trasformarlo in un segnale sugli edge tramite una trasformazione non lineare e propagarlo
    def get_incidences(self):

        B1 = self.boundaries[0].A
        B2 = self.boundaries[1].A

        B1_t = torch.tensor(B1).to_sparse().to(torch.float32).coalesce()
        B2_t = torch.tensor(B2).to_sparse().to(torch.float32).coalesce()

        return [B1_t, B2_t]

    def __getitem__(self, index):
        # Returns (xb, yb, mask) pair
        return self.Xe.float(), \
            self.y, self.mask

    def __len__(self):
        # Returns length
        return 1  # len(self.mask)


class FLOWDataset(torch.utils.data.Dataset):
    def __init__(self, data_path=r"data/flow/data.pkl",
                 incidence_data=r"data/flow/incidence.pkl",
                 split="train"):

        assert split in ["train", "val"]
        self.data = pickle.load(open(data_path, "rb"))
        self.data = self.data[split]
        self.data_dict = defaultdict(list)
        for X, y in self.data:
            self.data_dict['X'].append(X)
            self.data_dict['y'].append(y)

        self.data_dict['X'] = torch.stack(self.data_dict['X']).to(device)
        self.data_dict['y'] = torch.tensor(self.data_dict['y']).to(device)
        self.incicences = pickle.load(open(incidence_data, "rb"))

    def get_incidences(self):

        B1 = self.incicences["B1"]
        B2 = self.incicences["B2"]

        B1_t = torch.tensor(B1).to_sparse().to(torch.float32).coalesce()
        B2_t = torch.tensor(B2).to_sparse().to(torch.float32).coalesce()

        return [B1_t, B2_t]

    def __getitem__(self, index):
        # Returns (xb, yb) pair

        X = self.data_dict['X'][index]

        y = self.data_dict['y'][index]

        return X, y.squeeze(0)

    def __len__(self):
        # Returns length
        return len(self.data_dict['X'])


class CollaborationComplex(torch.utils.data.Dataset):
    def __init__(self, pct_miss, order, num_exp, eps, kappa,
                 device,
                 starting_node=150250,
                 data_path=r"data/collaboration_complex",):

        assert order >= 0
        assert pct_miss in range(10, 60, 10)
        self.incidences = np.load('{}/{}_boundaries.npy'.format(data_path,
                                                                starting_node),
                                  allow_pickle=True)

        # workaround order == len(self.incidences)
        # is not taken into account at the moment since is higher than
        # the maximum number used in the experiments.
        """
        Lup = torch.from_numpy(
            (self.incidences[order] @ self.incidences[order].T).A)
        if order == 0:
            Ldo = torch.zeros_like(Lup)
        else:
            Ldo = torch.from_numpy(
                (self.incidences[order-1].T @ self.incidences[order-1]).A)
        """
        Lup = np.load("{}/{}_laplacians_up.npy".format(data_path, starting_node), allow_pickle=True)[order]
        Ldo = np.load("{}/{}_laplacians_down.npy".format(data_path, starting_node), allow_pickle=True)[order]
        L = np.load("{}/{}_laplacians.npy".format(data_path, starting_node), allow_pickle=True)[order]

        Ldo = coo2tensor(normalize2(L, Ldo ,half_interval=True)).to_dense()
        Lup = coo2tensor(normalize2(L, Lup ,half_interval=True)).to_dense()
        L1 = coo2tensor(normalize2(L, L ,half_interval=True)).to_dense()

        self.L = (Ldo, Lup, compute_projection_matrix(
            L1, eps=eps, kappa=kappa))

        observed_signal = np.load('{}/{}_percentage_{}_input_damaged_{}.npy'.format(
            data_path, starting_node, pct_miss, num_exp), allow_pickle=True)
        observed_signal = [torch.tensor(
            list(signal.values()), dtype=torch.float) for signal in observed_signal]

        #
        target_signal = np.load(
            '{}/{}_cochains.npy'.format(data_path, starting_node), allow_pickle=True)
        target_signal = [torch.tensor(
            list(signal.values()), dtype=torch.float) for signal in target_signal]

        masks = np.load('{}/{}_percentage_{}_known_values_{}.npy'.format(data_path, starting_node, pct_miss,
                        num_exp), allow_pickle=True)  # positive mask= indices that we keep ##1 mask #entries 0 degree
        masks = [torch.tensor(
            list(mask.values()), dtype=torch.long) for mask in masks]

        self.X = observed_signal[order].reshape(-1,1).to(device)
        self.y = target_signal[order].to(device)
        self.n = len(self.X)
        self.mask = masks[order].to(device)

    def __getitem__(self, index):
        return self.X, self.y

    def __len__(self):
        # Returns length
        return 1000


class OCEANDataset(torch.utils.data.Dataset):
    def __init__(self, data_path=r"data/ocean/data.pkl",
                 incidence_data=r"data/ocean/incidence.pkl",
                 split="train"):

        assert split in ["train", "val", "test"]
        self.data = pickle.load(open(data_path, "rb"))
        self.data = self.data[split]
        self.data_dict = defaultdict(list)
        for X, y in self.data:
            self.data_dict['X'].append(X)
            self.data_dict['y'].append(y)

        self.data_dict['X'] = torch.stack(self.data_dict['X']).to(device)
        self.data_dict['y'] = torch.tensor(self.data_dict['y']).to(device)
        self.incicences = pickle.load(open(incidence_data, "rb"))

    def get_incidences(self):

        B1 = self.incicences["B1"]
        B2 = self.incicences["B2"]

        B1_t = torch.tensor(B1).to_sparse().to(torch.float32).coalesce()
        B2_t = torch.tensor(B2).to_sparse().to(torch.float32).coalesce()

        return [B1_t, B2_t]

    def __getitem__(self, index):
        # Returns (xb, yb) pair
        X = self.data_dict['X'][index]

        y = self.data_dict['y'][index]

        return X, y.squeeze(0)

    def __len__(self):
        # Returns length
        return len(self.data_dict['X'])
