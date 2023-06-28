import numpy as np
import argparse
import yaml
import os
import psutil

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
#from torchsummary import summary

from utils.data_loader import prepare_input_data
from utils.network_manipulation import train_net, resume_model
from networks.resnet_example import resnet18
from networks.preact_resnet import PreActResNet18
from networks.coatnet import CoAtNet



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

    parser.add_argument("--local-rank", type=int) #https://zhuanlan.zhihu.com/p/86441879  (torch.distributed.launch) -> borrowed from liu zhen
    
    args = parser.parse_args()

    ###############################################################
    # configure from yaml file
    config          = yaml_load(args.config)
    h5_path         = config['data']['h5file']
    csv_path        = config['data']['csvfile']
    loss_acc_path   = config['data']['loss_acc_path']
    loss_acc_file   = config['data']['loss_acc_file']
    modelpath       = config['data']['modelpath']
    modelfile       = config['data']['modelfile']
    input_shape     = [int(i) for i in config['data']['input_shape']]
    lr              = config['fit']['compile']['initial_lr']
    batch_size      = config['fit']['batch_size']
    epochs          = config['fit']['epochs']
    num_workers     = config['fit']['num_workers']
    model_name      = config['model']['name']

    print("Number of epochs: ", epochs)
    print("Max learning rate: ", lr)
    print("Batch size: ", batch_size)
    print("Number of features: ", input_shape[2])

    ###############################################################
    # configure device
    torch.cuda.is_available() 
    torch.cuda.empty_cache()
    torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    device_ids   = range(torch.cuda.device_count())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids).strip('[]').replace(' ', '')

    #----- GPU Parallel: (1) initialize and local_rank
    #https://pytorch.org/docs/stable/distributed.html
    #https://zhuanlan.zhihu.com/p/86441879
    #torch.distributed.init_process_group(backend="nccl")    
    #torch.distributed.init_process_group(backend="nccl")
    torch.distributed.init_process_group(backend="nccl", init_method='env://')         
    torch.cuda.set_device(args.local_rank)  # before your code runs
    device = torch.device("cuda", args.local_rank)
 
    start_epoch = 0

    ###############################################################
    torch.cuda.set_device(args.local_rank)  # before your code runs
    device = torch.device("cuda", args.local_rank)
 
    start_epoch = 0

    ###############################################################
    # preparing data
    print('\n===> Preparing dataset...')
    train_loader, validation_loader = prepare_input_data(h5_path, csv_path, input_shape[2], batch_size, num_workers=num_workers)
    print (f'memory usage： {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024 :.4f} GB' ) 
    print(f"batch_size = {batch_size}"  )
    #print(f"dataset_size = {dataset_size:d}»D"  )
    #print("dataset[0][0].size(): ",full_data[0][0].shape )    
    print("Length of the train_loader:", len(train_loader))
    print("Length of the val_loader:", len(validation_loader))

    ###############################################################
    # build network
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
    else:
        raise ValueError('Unknown network architecture.')


    net = net.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
        # GPU parallel:
        ## method 1: DataParallel
        ## net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
        ## method 2: DistributedDataParallel
        ## run with:  python -m torch.distributed.launch xxx.py
        #'''
        #https://zhuanlan.zhihu.com/p/86441879
        #https://www.cxyzjd.com/article/lbj23hao1/115691846    
        #https://www.zhihu.com/question/451269794/answer/1803183544
        
        #from torch.nn.parallel import DistributedDataParallel as DDP
        ##torch.distributed.init_process_group("nccl", init_method='env://')     
        #torch.distributed.init_process_group(backend="nccl")

        net = DDP(net, find_unused_parameters=True, device_ids=[args.local_rank], output_device=args.local_rank) 
        #if args.local_rank == 0:
        #    summary(net, input_shape=(2, 250, 250))

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # define optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-08, amsgrad=False)
        

    ## If the action is resume, then load previous trained model :
    if args.resume and os.path.exists(modelpath + modelfile):
        net, best_acc, start_epoch = resume_model(net, device, modelpath, modelfile)


    ###############################################################
    # traing network
    print(f"\n===> Training network {model_name}...")

    train_net(start_epoch, epochs, device, lr, net, criterion, optimizer, train_loader, validation_loader, args.resume, args.save_all, loss_acc_path, loss_acc_file, modelpath, modelfile)






