import h5py as h5
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler

class StripData(data.Dataset):
    def __init__(self, h5_path, csv_path, n_channels=2):
        csv_info        = pd.read_csv(csv_path, header=None)
        self.groupname  = np.asarray(csv_info.iloc[:, 0])
        self.datainfo   = np.asarray(csv_info.iloc[:, 1])
        self.h5file     = h5.File(h5_path, 'r')
        self.n_channels = n_channels

    def __len__(self):
        return len(self.datainfo)

    def __getitem__(self, idx):
        dset_entry = self.h5file[self.groupname[idx]][self.datainfo[idx]]
        eventtype = 1 if "bb0n" in self.datainfo[idx] else 0 
        img = np.array(dset_entry)[:, :, :self.n_channels]
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).type(torch.FloatTensor), eventtype



def prepare_input_data(h5_path, csv_path, n_channels, batch_size, validation_split=0.2):
    nEXODataset = StripData(h5_path, csv_path, n_channels)
    datasize = len(nEXODataset)
    print(f"Total data sample size = {datasize}, where {1-validation_split} for training and {validation_split} for validation.")

    # shuffling dataset:
    indices = list(range(datasize))
    split = int(np.floor(validation_split * datasize))
    shuffle_dataset = True
    random_seed = 61
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PyTorch data samplers and loaders
    train_sampler =  SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=batch_size, sampler=validation_sampler, num_workers=0)

    return train_loader, validation_loader

