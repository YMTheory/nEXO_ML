#!/usr/bin/env python
'''Merge hdf5 files into a single file'''
'''Also generate a csv file including group and dataset names.'''
import h5py    # HDF5 support
import os
import glob
import time
import csv
import argparse
import pandas as pd

def h5merger(filedir, c_file, h5_file):
    filelist = glob.glob('%s/*.h5' % filedir)
    csvfile = open(c_file, 'w')
    fieldnames = ['groupname', 'dsetname']
    writer = csv.DictWriter(csvfile, fieldnames)
    with h5py.File(h5_file, 'w') as fid:
        for i in range(len(filelist)):
            fileName = filelist[i]
            if "gamma" in fileName:
                groupname = 'gamma'
            else:
                groupname = 'bb0n'
            print(fileName, groupname)
            f = h5py.File(fileName,  "r")
            f.copy(f[groupname], fid['/'], name='nexo_data_%d' % i)
            dset = f[groupname]
            for item in dset.keys():
                writer.writerow({'groupname':'nexo_data_%d' % i, 'dsetname':item})
            f.close()
    csvfile.close() 
    #csv_info = pd.read_csv(c_file, header=None, delimiter=',')
    #shuffled = csv_info.sample(frac=1).reset_index()
    #n_train = int(len(shuffled)*0.8)
    #shuffled = csv_info.sample(frac=1).reset_index()
    #shuffled[:n_train].to_csv(c_file.replace('nexo', 'nexo_train'), index=False, columns =[0, 1])
    #shuffled[n_train:].to_csv(c_file.replace('nexo', 'nexo_valid'), index=False, columns =[0, 1])




def shuffling(csvfile):
    csv_info = pd.read_csv(csvfile, header=None, delimiter=',')
    shuffled = csv_info.sample(frac=1).reset_index()
    n_train = int(len(shuffled)*0.8)
    shuffled = csv_info.sample(frac=1).reset_index()
    
    cfile = 'test_train.csv'
    shuffled_train = shuffled[:n_train].replace({'nexo': 'nexo_train'})
    shuffled_train.to_csv(cfile, index=False, columns=[0, 1], mode='a')

    cfile = 'test_valid.csv'
    shuffled_valid = shuffled[n_train:].replace({'nexo': 'nexo_valid'})
    shuffled_valid.to_csv(cfile, index=False, columns=[0, 1], mode='a')



def totalSampleCounts(particle):
    path = "/hpcfs/juno/junogpu/miaoyu/bb0n/dataset/strip_6mm/lzp/"
    st = f"nexo_{particle}"
    files = [d for d in os.listdir(path) if  d.startswith(st)]
    count = 0
    for f in files:
        with h5py.File(path+f, "r") as fin:
            group_name = particle
            g = fin[group_name]
            count += len(g.keys())

    print(f"Total entries for {particle} training data is {count}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset builder.')
    parser.add_argument('--filedir', '-f', type=str, help='directory of h5 files.')
    parser.add_argument('--outfile', '-o', type=str, help='output h5 file.')
    parser.add_argument('--csvfile', '-c', type=str, help='csv file of dataset info.')
    args = parser.parse_args()
    h5merger(args.filedir, args.csvfile, args.outfile)
    #shuffling(args.csvfile)
    #totalSampleCounts('bb0n')
