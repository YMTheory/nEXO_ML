import h5py as h5
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms


class StripData(data.Dataset):
    def __init__(self, h5_path, csv_path, n_channels=2, process=False):
        csv_info        = pd.read_csv(csv_path, header=None)
        self.groupname  = np.asarray(csv_info.iloc[:, 0])
        self.datainfo   = np.asarray(csv_info.iloc[:, 1])
        self.h5file     = h5.File(h5_path, 'r')
        self.n_channels = n_channels
        self.process    = process

    def __len__(self):
        return len(self.datainfo)

    def __getitem__(self, idx):
        dset_entry = self.h5file[self.groupname[idx]][self.datainfo[idx]]
        #eventtype = 1 if "bb0n" in self.datainfo[idx] else 0  # labelling for my data format
        eventtype = 1 if dset_entry.attrs[u'tag']=='e-' else 0 # for zepeng's data format:
        #img = np.array(dset_entry)[:, :self.n_channels, :]
        #img = np.transpose(img, (1, 2, 0)) # for my data
        img = np.array(dset_entry[:, :, :self.n_channels])
        img = np.transpose(img, (2, 0, 1)) # for zepeng's data
        print("numpy array shape: ", img.shape)
        ##img[img==-100] = 0.
        # preprocessing ##############
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
        ##bkg = torch.randn(img_tensor.shape) * 200
        ##img_tensor = img_tensor + bkg
        ##preprocess = transforms.Compose([transforms.Resize((256, 256)),  transforms.Normalize(means, stds) ])
        if self.process:
            preprocess = transforms.Resize((256, 256))
            img_tensor = preprocess(img_tensor) 
        ##############################
        return img_tensor, eventtype



class StripBigData(data.Dataset):
    def __init__(self, h5_path, csv_file, n_channels=2):
        csv_info        = pd.read_csv(csv_file, header=None)
        self.h5_list    = np.asarray(csv_info.iloc[:, 0])
        self.groupname  = np.asarray(csv_info.iloc[:, 1])
        self.datainfo   = np.asarray(csv_info.iloc[:, 2])
        self.n_channels = n_channels

    def __len__(self):
        return len(self.datainfo)

    def __getitem__(self, idx):
        fileName = self.h5_list[idx]
        print(self.h5_list[idx], self.groupname[idx], self.datainfo[idx])
        with h5.File(fileName, 'r') as h5_file:
            dset_entry = h5_file[self.groupname[idx]][self.datainfo[idx]]
        eventtype = 1 if "bb0n" in self.datainfo[idx] else 0 
        img = np.array(dset_entry)[:, :, :self.n_channels]
        img = np.transpose(img, (2, 0, 1))
        img[img==-100] = 0.
        preprocess = transforms.Resize((256, 256))
        img_tensor = preprocess(img_tensor) 
        return img_tensor, eventtype



def prepare_input_data(h5_path, csv_path, n_channels, batch_size, dtb=True, num_workers=0, validation_split=0.2):
    nEXODataset = StripData(h5_path, csv_path, n_channels)
    datasize = len(nEXODataset)
    print(f"Total data sample size = {datasize}, where {1-validation_split} for training and {validation_split} for validation.")

    if not dtb:
        # method 1
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
        train_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=batch_size, sampler=validation_sampler, num_workers=num_workers)

        return train_loader, valid_loader

    else:
        # method 2: DistributedSampler
        train_size = int((1-validation_split) * datasize)
        valid_size = int(datasize - train_size)

        train_dataset, valid_dataset = random_split(nEXODataset, [train_size, valid_size])
        #train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
        #                          num_workers=n_workers, shuffle=True, pin_memory=True)
        #----- GPU Parallel: (3)-1 DistributedSampler 
        #                    https://github.com/laisimiao/classification-cifar10-pytorch/blob/master/main_ddp.py
        #                BatchSampler  https://www.zhihu.com/search?type=content&q=SyncBatchNorm
        sampler = DistributedSampler(train_dataset)  
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        train_batch_sampler=torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
        
        train_loader = DataLoader(train_dataset,  batch_sampler=train_batch_sampler, 
                              num_workers=num_workers, shuffle=False, pin_memory=True)  # shuffle can not be specified as True in this case.
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size,  sampler=test_sampler,
                             num_workers=num_workers, shuffle=False, pin_memory=True, drop_last=True)

        return train_loader, valid_loader

