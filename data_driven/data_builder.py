# data_driven/data_builder.py

from torch.utils.data import Dataset
import torch
import numpy as np

class Dataset_MSL(Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Dataset_MSL implementation is not included in the open-source version.")
    
    def __getitem__(self, index):
        return None

    def __len__(self):
        return 0


class Dataset_SMAP(Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Dataset_SMAP implementation is not included in the open-source version.")
    
    def __getitem__(self, index):
        return None

    def __len__(self):
        return 0


class Dataset_SMD(Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Dataset_SMD")
    
    def __getitem__(self, index):
        return None

    def __len__(self):
        return 0


class Dataset_SWaT(Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Dataset_SWaT")
    
    def __getitem__(self, index):
        return None

    def __len__(self):
        return 0


class MultiDomainDataset(Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("MultiDomainDataset")

    def __getitem__(self, index):
        return None

    def __len__(self):
        return 0

    def get_domain_info(self):
        return []
