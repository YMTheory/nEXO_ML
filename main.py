import numpy as np
import argparse
import yaml
import os

import torch
import torch.nn as nn

from utils.data_loader import prepare_input_data
from utils.network_manipulation import train_net, resume_model
from networks.resnet_example import resnet18
from networks.preact_resnet import PreActResNet18
from networks.coatnet import CoAtNet


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

list_of_gpus = range(torch.cuda.device_count())
device_ids = range(torch.cuda.device_count())
os.environ["CUDA_VISIBLE_DEVICES"] = str(list_of_gpus).strip('[]').replace(' ', '')

start_epoch = 0


def yaml_load(config):
    """
    Load configurations from yaml file.
    """
    with open(config) as stream:
        param = yaml.safe_load(stream)
        return param



if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='PyTorch nEXO background rejection')
    parser.add_argument('--config',     '-f', type=str,  default="baseline.yml",  help="specify yaml config")
    parser.add_argument('--resume',     dest="resume",   action='store_true',     help='resume from checkpoint')
    parser.add_argument('--no-resume',  dest="resume",   action='store_false',    help='not resume from checkpoint')
    parser.add_argument('--save',       dest="save_all", action='store_true',     help='save all outputs')
    parser.add_argument('--no-save',    dest="save_all", action='store_false',    help='not save all outputs')
    
    args = parser.parse_args()

    config = yaml_load(args.config)
    h5_path = config['data']['h5file']
    csv_path = config['data']['csvfile']
    loss_acc_path = config['data']['loss_acc_path']
    loss_acc_file = config['data']['loss_acc_file']
    modelpath = config['data']['modelpath']
    modelfile = config['data']['modelfile']
    input_shape = [int(i) for i in config['data']['input_shape']]
    lr = config['fit']['compile']['initial_lr']
    batch_size = config['fit']['batch_size']
    epochs = config['fit']['epochs']
    model_name = config['model']['name']


    # preparing data
    print('\n===> Preparing dataset...')
    train_loader, validation_loader = prepare_input_data(h5_path, csv_path, input_shape[2], batch_size, dtb=False)
    print("Length of the train_loader:", len(train_loader))
    print("Length of the val_loader:", len(validation_loader))

    print(f'\n===> Buiding model {model_name}...')
    if model_name == "resnet18":
        net = resnet18(input_channels=input_shape[2])
    elif model_name == "preact_resnet18":
        net = PreActResNet18(num_channels=input_shape[2])
    elif model_name == "coatnet":
        num_blocks = [2, 2, 6, 14, 2]           # L
        channels = [128, 128, 256, 512, 1026]   # D
        Nlabels = 2
        net = CoAtNet((256, 256), 2, num_blocks, channels, num_classes=Nlabels)        

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-08, amsgrad=False)

    net = net.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
        net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
        

    ## If the action is resume, then load previous trained model :
    if args.resume and os.path.exists(modelpath + modelfile):
        net, best_acc, start_epoch = resume_model(net, device, modelpath, modelfile)

    # traing network
    print(f"\n===> Training network {model_name}...")

    train_net(start_epoch, epochs, device, lr, net, criterion, optimizer, train_loader, validation_loader, args.resume, args.save_all, loss_acc_path, loss_acc_file, modelpath, modelfile)






