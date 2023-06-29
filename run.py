#############################################################################
''' 
2023.06.29
Modeified from Zhen's script: /hpcfs/juno/junogpu/miaoyu/ml_atmospheric_neutrino/rec_dir_xyz/run.py
Using torch.DistributedDataParallel to train the network in multiple GPUs.

'''
#############################################################################


# load basic modules
import argparse
import numpy as np
import time
import os
import yaml
import psutil #memory 

# load torch-related modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
#from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
#from torchsummary import  summary

# load utils
from utils.data_loader import StripData
from utils.data_loader import prepare_input_data
#from utils.network_manipulation import train_net, resume_model
from utils.network_manipulation_parallel import train, validation

# load networks
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



def main():
    parser = argparse.ArgumentParser(description='PyTorch nEXO PID task.')
    
    parser.add_argument("--local-rank", type=int) #https://zhuanlan.zhihu.com/p/86441879  (torch.distributed.launch)
    
    parser.add_argument('--config',     '-f', type=str,  default="baseline.yml",  help="specify yaml config")
    #parser.add_argument('--resume',     dest="resume",   action='store_true',     help='resume from checkpoint')
    #parser.add_argument('--no-resume',  dest="resume",   action='store_false',    help='not resume from checkpoint')

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
    train_split     = config['data']['train_split']
    INPUT_SHAPE     = [int(i) for i in config['data']['input_shape']]
    MAX_LR          = config['fit']['compile']['initial_lr']
    BATCH_SIZE      = config['fit']['batch_size']
    EPOCHS          = config['fit']['epochs']
    NUM_WORKERS     = config['fit']['num_workers']
    WEIGHT_DECAY    = config['fit']['compile']['weight_decay']
    DIV_Factor      = config['fit']['compile']['DIV_factor']
    GRAD_CLIP       = config['fit']['compile']['grad_clip']
    LOG_INTERVAL    = config['fit']['log_interval']
    model_name      = config['model']['name']
    loss_func       = config['fit']['compile']['loss']
    optim_name      = config['fit']['compile']['optimizer']

    print("Number of epochs: ",     EPOCHS)
    print("Max learning rate: ",    MAX_LR)
    print("Batch size: ",           BATCH_SIZE)
    print("Number of features: ",   INPUT_SHAPE[2])
    print("Weight decay: ",         WEIGHT_DECAY)
    print("Gradient clip: ",        GRAD_CLIP)


    ###############################################################
    # configure device

    torch.cuda.is_available() 
    torch.cuda.empty_cache()
    torch.cuda.device_count()
    # Device configuration
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #----- GPU Parallel: (1) initialize and local_rank
    #https://pytorch.org/docs/stable/distributed.html
    #https://zhuanlan.zhihu.com/p/86441879
    #torch.distributed.init_process_group(backend="nccl")    
    torch.distributed.init_process_group(backend="nccl", init_method='env://')         
    torch.cuda.set_device(args.local_rank)  # before your code runs
    device = torch.device("cuda", args.local_rank)
 
 
    ###############################################################
    # prepare data loader
    full_data = StripData(h5_path, csv_path, INPUT_SHAPE[2])
    dataset_size = len(full_data)

    train_size = int(len(full_data)*train_split)  
    test_size = len(full_data) - train_size     

    train_dataset, test_dataset =random_split(full_data, [train_size, test_size])

    #train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
    #                          num_workers=n_workers, shuffle=True, pin_memory=True)
    #----- GPU Parallel: (3)-1 DistributedSampler 
    #                    https://github.com/laisimiao/classification-cifar10-pytorch/blob/master/main_ddp.py
    #                BatchSampler  https://www.zhihu.com/search?type=content&q=SyncBatchNorm
    sampler = DistributedSampler(train_dataset)  
    train_sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler=torch.utils.data.distributed.DistributedSampler(test_dataset)
    train_batch_sampler=torch.utils.data.BatchSampler(train_sampler,BATCH_SIZE,drop_last=True)
    
    train_loader = DataLoader(train_dataset,  batch_sampler=train_batch_sampler, 
                          num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,  sampler=test_sampler,
                         num_workers=NUM_WORKERS, shuffle=False, pin_memory=True, drop_last=True)
    print (f'memory usage： {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024 :.4f} GB' ) 
    print(f"batch_size = {BATCH_SIZE:d}"  )
    print(f"dataset_size = {dataset_size:d}»D"  )
    print("dataset[0][0].size(): ",full_data[0][0].shape )    
    print("Length of the train_loader:", len(train_loader))
    print("Length of the val_loader:", len(test_loader))
    ##images, labels = next(iter(test_loader))
    ##print("images.shape: ",images.shape)
    ##print("labels.shape: ",labels.shape)
    ##print(labels)


    ###############################################################
    # build network
    print(f'\n===> Buiding model {model_name}...')
    if model_name == "resnet18":
        net = resnet18(input_channels=INPUT_SHAPE[2])
    elif model_name == "preact_resnet18":
        net = PreActResNet18(num_channels=INPUT_SHAPE[2])
    elif model_name == "coatnet":
        num_blocks = [2, 2, 6, 14, 2]           # L
        channels = [128, 128, 256, 512, 1026]   # D
        Nlabels = 2
        net = CoAtNet((256, 256), 2, num_blocks, channels, num_classes=Nlabels)        
    else:
        raise ValueError('Unknown network architecture.')
          
        
    #####=========================================
    ## GPU Parallel
    ## method 1: DataParallel
    '''    
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)        
    model.to(device)    
    '''

    ## method 2: DistributedDataParallel
    ## run with:  python -m torch.distributed.launch xxx.py
    #'''
    #https://zhuanlan.zhihu.com/p/86441879
    #https://www.cxyzjd.com/article/lbj23hao1/115691846    
    #https://www.zhihu.com/question/451269794/answer/1803183544
    
    #from torch.nn.parallel import DistributedDataParallel as DDP
    ##torch.distributed.init_process_group("nccl", init_method='env://')     
    #torch.distributed.init_process_group(backend="nccl")
    
    #----- GPU Parallel: (4) DDP
    net.to(device)
    if torch.cuda.device_count() > 1:    
        #model = DDP(model)
        net = DDP(net, find_unused_parameters=True, device_ids=[args.local_rank], output_device=args.local_rank)    
    #'''
    #if args.local_rank == 0:        
    #    summary(net, input_size=(INPUT_SHAPE[2], INPUT_SHAPE[0], INPUT_SHAPE[1]))
    
    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()
    print("Loss function: ", loss_func)

    # define optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=MAX_LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY, eps=1e-08, amsgrad=False)
    print('Optimizer name: ', optim_name)

    # define learning rate scheduler
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, final_div_factor=DIV_Factor, epochs=EPOCHS, steps_per_epoch=len(train_loader))
        
    
    # 22.04.14 https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/
    #  Automatic Mixed Precision (AMP)
    #  https://blog.csdn.net/qq_21539375/article/details/117231128
    scaler = torch.cuda.amp.GradScaler() 
    
 
    print (f'memory usage： {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024 :.4f} GB' ) 
    
    # record training process
    epoch_train_losses = []
    epoch_test_losses = []
    epoch_train_acc = []
    epoch_test_acc = []
    epoch_best = -1
    epoch_best_loss = 1e2
    
    epoch_Learning_Rates = []
        
    # start training
    for epoch in range(EPOCHS):
        start =time.time()
        #----- GPU Parallel: (3)-2 DistributedSampler 
        sampler.set_epoch(epoch)
        
        # train, test model
        #train_losses, train_acc, Learning_rate      = train(LOG_INTERVAL, net, device, train_loader, optimizer, criterion, epoch, EPOCHS, scheduler, scaler, grad_clip=GRAD_CLIP)
        #test_losses, test_acc                       = validation(net, device, optimizer, criterion, test_loader)
        train_losses, Learning_rate      = train(LOG_INTERVAL, net, device, train_loader, optimizer, criterion, epoch, EPOCHS, scheduler, scaler, grad_clip=GRAD_CLIP)
        test_losses                      = validation(net, device, optimizer, criterion, test_loader)
        
        #scheduler.step()        

        if args.local_rank == 0:        
            if(test_losses<epoch_best_loss) :
                epoch_best_loss=test_losses
                epoch_best=epoch
                #https://discuss.pytorch.org/t/how-to-save-the-best-model/84608 
                #----- GPU Parallel: (1) local_rank
                torch.save(net.state_dict(), os.path.join(modelpath, model_name+"_best-model0-parameters.pt"))  
                torch.save(net.module.state_dict(), os.path.join(modelpath, model_name+"_best-model-parameters.pt"))
            # save results
            epoch_Learning_Rates.append(Learning_rate)
    
            epoch_train_losses.append(train_losses)
            epoch_test_losses.append(test_losses)
            #epoch_train_acc.append(train_acc)
            #epoch_test_acc.append(test_acc)

    
        end = time.time()
        print('\n Running time: %s min for epoch %d'%((end-start)/60, epoch))    

    if args.local_rank == 0:             
        print(f'epoch_best_loss = {epoch_best_loss:.6f},  GPU#{args.local_rank}: epoch_best = {epoch_best:.0f}, lr_best = {epoch_Learning_Rates[epoch_best][-1]:.2E}')    
        print("DONE!")    
        # save all train test results
        LR = np.array(epoch_Learning_Rates)    
        A = np.array(epoch_train_losses)
        B = np.array(epoch_test_losses)
        #C = np.array(epoch_train_acc)
        #D = np.array(epoch_test_acc)
        np.save(os.path.join(loss_acc_path, model_name+"_T"+str(len(train_loader))+"_epoch_Learning_Rates.npy" ), LR)    
        np.save(os.path.join(loss_acc_path, model_name+"_N"+str(EPOCHS)+"_Ntrain_losses.npy" ), A)
        np.save(os.path.join(loss_acc_path, model_name+"_N"+str(EPOCHS)+"_Ntest_losses.npy" ),  B)
        #np.save(os.path.join(loss_acc_path, model_name+"_N"+str(EPOCHS)+"_Ntrain_acc.npy" ),    C)
        #np.save(os.path.join(loss_acc_path, model_name+"_N"+str(EPOCHS)+"_Ntest_acc.npy" ),     D)
        print("SAVED!") 
    
    
main()
           
