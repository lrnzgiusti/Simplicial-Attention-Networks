#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Lorenzo Giusti
"""


# Importing necessary libraries and modules
from .utils import normalize, compute_projection_matrix, normalize2, coo2tensor
import torch
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict


class SyntheticVectorField(torch.utils.data.Dataset):
    """
    Class for handling the Synthetic Vector Field dataset.
    Inherits from torch.utils.data.Dataset.
    This dataset is used for a specific type of machine learning task, and this class handles the data manipulation and
    formatting for that task.
    """

    def __init__(self, data_path=r"data/vec_ad/synth_vec_ad.pkl", split="train", device="cpu"):
        """
        Constructor of the SyntheticVectorField class.

        :param data_path: string, path where the data file is located. Default is "data/vec_ad/synth_vec_ad.pkl".
        :param split: string, specifies if the dataset is for training or testing. Default is "train".
        :param device: string, specifies the device where the tensors should be allocated. Default is "cpu".
        """

        # Load the pickled data
        self.data = pickle.load(open(data_path, "rb"))

        # Convert the data to torch tensors
        self.X = torch.from_numpy(self.data['X']).float()
        self.y = torch.from_numpy(self.data['y']).long()

        # Create an index to split the data into training and testing sets
        idx = int(len(self.X)*0.8)

        # If it is a training split, take the first 80% of the data
        if split == "train":
            self.X = self.X[:idx].to(device)
            self.y = self.y[:idx].to(device)
        # If it is not a training split, take the remaining 20% of the data
        else:
            self.X = self.X[idx:].to(device)
            self.y = self.y[idx:].to(device)

        # stored in scipy.coo_matrix
        self.boundaries = [self.data['B1'], self.data['B2']]

    def get_incidences(self):
        """
        Method to get the incidences of the boundaries in tensor format.

        :return: list, tensors of the incidences.
        """

        # Get the boundary matrices
        B1 = self.boundaries[0].A
        B2 = self.boundaries[1].A

        # Convert the boundary matrices to sparse tensors
        B1_t = torch.tensor(B1).to_sparse().to(torch.float32).coalesce()
        B2_t = torch.tensor(B2).to_sparse().to(torch.float32).coalesce()

        # Return the tensors
        return [B1_t, B2_t]

    def __getitem__(self, index):
        """
        Method to get an item from the dataset given an index.

        :param index: int, index of the desired item.
        :return: tuple, (X, y) pair.
        """
        X, y = self.X[index], self.y[index]
        return X, y[0]

    def __len__(self):
        """
        Method to get the length of the dataset.

        :return: int, length of the dataset.
        """
        return len(self.X)

class SyntheticSIG(torch.utils.data.Dataset):
    """
    SyntheticSIG is a Dataset class that loads and manages synthetic signal data for model training and validation.
    
    Args:
        data_path (str): The file path to the pickle file containing the synthetic signal data.
        split (str): Determines whether the data loaded is for training or validation. Options are 'train' or 'test'.
        device (str): The device where the tensors will be allocated.
    
    Attributes:
        data (dict): The data loaded from the pickle file.
        X (tensor): Input features tensor.
        y (tensor): Target labels tensor.
        boundaries (list): A list containing the boundaries of the data.
    """

    def __init__(self, data_path=r"data/synth/synth_X_Y.pkl", split="train", device="cpu"):
        self.data = pickle.load(open(data_path, "rb"))  # Load the data from the pickle file

        # Convert the numpy arrays to PyTorch tensors and transpose them to match the expected input shape
        self.X = torch.from_numpy(self.data['X']).T.float().to(device)  
        self.y = torch.from_numpy(self.data['Y']).T.long().to(device)
        
        # Split data into training and testing based on the 'split' argument
        idx = int(len(self.X)*0.8)
        if split == "train":
            self.X = self.X[:idx]
            self.y = self.y[:idx]
        else:
            self.X = self.X[idx:]
            self.y = self.y[idx:]

        # Boundaries are stored in scipy.coo_matrix form
        self.boundaries = [self.data['B1'], self.data['B2']]

    def get_incidences(self):
        """
        Converts the boundary matrices from the sparse numpy format to sparse PyTorch tensors.

        Returns:
            list: A list containing the converted boundary matrices.
        """
        B1 = self.boundaries[0]
        B2 = self.boundaries[1]

        # Convert the numpy arrays to sparse PyTorch tensors
        B1_t = torch.tensor(B1).to_sparse().to(torch.float32).coalesce()
        B2_t = torch.tensor(B2).to_sparse().to(torch.float32).coalesce()

        return [B1_t, B2_t]

    def __getitem__(self, index):
        """
        Gets a data sample for a given index.

        Args:
            index (int): Index of the data sample.

        Returns:
            tuple: Tuple containing input features and the corresponding label for the given index.
        """
        X, y = self.X[index], self.y[index]
        return X, y[0]  # 'y' is a 1-D tensor, we return the scalar value

    def __len__(self):
        """
        Gets the total number of data samples.

        Returns:
            int: Total number of data samples.
        """
        return len(self.X)



class RNADataset(torch.utils.data.Dataset):
    """
    RNADataset is a Dataset class that loads and manages RNA data for model training and validation.
    
    Args:
        data_path (str): The file path to the pickle file containing the RNA data.
        split (str): Determines whether the data loaded is for training or validation. Options are 'train' or 'val'.
        store_boundaries (bool): Flag indicating whether to store boundaries or not.
        
    Attributes:
        data (dict): The data loaded from the pickle file.
        y (tensor): Target labels tensor.
        Xe (tensor): Edge features tensor.
        Xn (tensor): Node features tensor.
        boundaries (dict): A dictionary containing the boundaries of the data.
        train_mask (array): An array of indices for training data samples.
        valid_mask (array): An array of indices for validation data samples.
        mask (tensor): Tensor of selected indices based on the split type.
    """
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
    """
    FLOWDataset is a Dataset class that loads and manages flow data for model training and validation.

    Args:
        data_path (str): The file path to the pickle file containing the flow data.
        incidence_data (str): The file path to the pickle file containing the incidence data.
        split (str): Determines whether the data loaded is for training or validation. Options are 'train' or 'val'.

    Attributes:
        data (dict): The data loaded from the pickle file.
        data_dict (dict): A dictionary holding tensors of feature and target data.
        incidences (dict): The incidences data loaded from the pickle file.
    """
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
    """
    CollaborationComplex is a Dataset class that manages data from a collaboration complex.

    Args:
        pct_miss (int): Percentage of missing data.
        order (int): The order of the data.
        num_exp (int): Number of experiments.
        eps (float): A small value used for computing the projection matrix.
        kappa (float): A value used for computing the projection matrix.
        device (str): The device where the tensors will be allocated.
        starting_node (int): The starting node in the collaboration complex.
        data_path (str): The directory path to the collaboration complex data.
    
    Attributes:
        incidences (ndarray): Array of incidences.
        L (tuple): Tuple containing up, down, and projection matrices.
        X (tensor): Input features tensor.
        y (tensor): Target labels tensor.
        n (int): Length of the X tensor.
        mask (tensor): Mask tensor.
    """
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
        return len(self.X)


class OCEANDataset(torch.utils.data.Dataset):
    """
    OCEANDataset is a Dataset class that loads and manages OCEAN data for model training, validation, and testing.

    Args:
        data_path (str): The file path to the pickle file containing the OCEAN data.
        incidence_data (str): The file path to the pickle file containing the incidence data.
        split (str): Determines whether the data loaded is for training, validation, or testing. Options are 'train', 'val', or 'test'.

    Attributes:
        data (dict): The data loaded from the pickle file.
        data_dict (dict): A dictionary holding tensors of feature and target data.
        incidences (dict): The incidences data loaded from the pickle file.
    """
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
