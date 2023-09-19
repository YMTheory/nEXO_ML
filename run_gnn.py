from networks.gcn_example import MoNet, SimpleGCN
from networks.gravnet import PhotonNet
from utils.gnnnetwork_manipulation_parallel import train_gcn, validation_gcn, test_accuracy_gcn
from utils.graph_loader import build_graph

import torch
from torch_geometric.utils import degree
from torch_geometric.data import DataLoader
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.nn import summary
#from torch_geometric.transforms import NormalizeFeatures, NormalizeEdgeAttr


import time
import random
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='PyTorch nEXO PID task (GCN network).')
    
    parser.add_argument("--local-rank", type=int) #https://zhuanlan.zhihu.com/p/86441879  (torch.distributed.launch)

    parser.add_argument('--mode',   type=str, default='train', help='train or test mode.')
    parser.add_argument('--suffix', type=str, default='test', help='suffix for pt and npy files.')
    parser.add_argument('--model',  type=str, default='GravNet', help='Model name for GNN.')

    parser.add_argument('--resume',     dest="resume",   action='store_true',     help='resume from checkpoint')
    parser.add_argument('--no-resume',  dest="resume",   action='store_false',    help='not resume from checkpoint')

    parser.add_argument('--lr',          type=float, default=1e-6,   help='Constant learning rate or maximum value for the scheduled one.')
    parser.add_argument('--batch_size',  type=int,   default=64,     help='Batch size.')
    parser.add_argument('--kernel_size', type=int,   default=25,     help='Kernel size.')
    parser.add_argument('--nepoch',      type=int,   default=30,     help='Number of training epoch.')

    args = parser.parse_args()

    #full_data = GraphDataset('/hpcfs/juno/junogpu/miaoyu/bb0n/dataset/strip_6mm/graphs/')
    gpu_graph_path = '/hpcfs/juno/junogpu/miaoyu/bb0n/dataset/strip_6mm/graphs/'
    if args.model == "GravNet":
        filelist = [gpu_graph_path + 'nexo_bb0n_graph.h5', gpu_graph_path + 'nexo_gamma_graph.h5']
    elif args.model == "MoNet":
        filelist = [gpu_graph_path + 'nexo_bb0n_graph_monet.h5', gpu_graph_path + 'nexo_gamma_graph_monet.h5']
    full_data = build_graph(filelist)
    dataset_size = len(full_data)
    print(f"Total data size is {dataset_size}.")

    random.shuffle(full_data)
    split_ratio = 0.8
    train_size = int(split_ratio * dataset_size)
    train_dataset = full_data[:train_size]
    test_dataset  = full_data[train_size:]
    print(f'Total training data sample is {len(train_dataset)} and total test data sample is {len(test_dataset)}.')

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # config device
    torch.cuda.is_available()
    torch.cuda.empty_cache()
    torch.cuda.device_count()
    
    torch.distributed.init_process_group(backend="nccl", init_method='env://')         
    torch.cuda.set_device(args.local_rank)  # before your code runs
    device = torch.device("cuda", args.local_rank)
    
    print(f'Device avaliable is {device}.')
    
    # build model
    kernel_size = args.kernel_size
    #model = MoNet(kernel_size).to(device)
    
    onedata = train_dataset[0]
    num_node_features = onedata.num_node_features
    print('Number of node features: ', num_node_features)
    #hidden_channels = 64
    num_classes = 2
    
    if args.model == 'MoNet':
        model = MoNet(kernel_size, num_node_features, num_classes)

        if args.resume:
            resume_model = '/hpcfs/juno/junogpu/miaoyu/bb0n/jobs/MoNet_model_batchsize64_5.0e-04constLR_0801.pt'
            checkpoint = torch.load(resume_model)
            model.load_state_dict(checkpoint)
            #loaded_model.load_state_dict(torch.load('path/to/model.pt'))
    
        print('Building MoNet network finished...')
    elif args.model == 'GravNet':
        input_dim = 4
        model = PhotonNet(
                    input_dim = input_dim,
                    output_dim = 2,
                    grav_dim = 64,      #128,
                    hidden_dim = 256,   #256,
                    n_gravnet_blocks = 3,
                    n_postgn_dense_blocks = 3,
                    dropout = 0.5
                )
        print('Building GravNet network finished...')

    elif args.model == 'simple':
        hidden_channels = 128
        model = SimpleGCN(num_node_features, hidden_channels, num_classes)

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = DDP(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=args.local_rank)

    for name, param in model.named_parameters():
        print(name, param.shape)

    
    if args.mode == 'train':
        lr = args.lr
        weight_decay = 1e-5
        #decay_step = 1
        #lr_decay = 0.99
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_step, lr_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay, eps=1e-08, amsgrad=False)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        scaler = torch.cuda.amp.GradScaler() 
        GRAD_CLIP = 0.3

        NUM_EPOCH = args.nepoch
        train_losses_array,valid_losses_array = [], []
        print(f'Total training epoch is {NUM_EPOCH}.')
        for epoch in range(NUM_EPOCH):
            LOG_INTERVAL = 100
            train_losses = train_gcn(LOG_INTERVAL, model, device, train_loader, optimizer, criterion, epoch, NUM_EPOCH, scaler, GRAD_CLIP, args.model) # 0719
            valid_losses = validation_gcn(model, device, optimizer, criterion, test_loader, args.model)
            print(f'Epoch {epoch}: training loss is {train_losses}, validation loss is {valid_losses}.')
            train_losses_array.append(train_losses)
            valid_losses_array.append(valid_losses)

            #train_acc = test(train_loader)
            #test_acc  = test(test_loader)
            #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        tla = np.array(train_losses_array)
        np.save(f'/hpcfs/juno/junogpu/miaoyu/bb0n/nEXO_ML/training_outputs/{args.model}_batchsize{batch_size}_{lr:.1e}constLR_trainloss_{args.suffix}.npy', tla)
        tla = np.array(valid_losses_array)
        np.save(f'/hpcfs/juno/junogpu/miaoyu/bb0n/nEXO_ML/training_outputs/{args.model}_batchsize{batch_size}_{lr:.1e}constLR_validloss_{args.suffix}.npy', tla)
        torch.save(model.module.state_dict(), f'/hpcfs/juno/junogpu/miaoyu/bb0n/nEXO_ML/checkpoint_sens/{args.model}/{args.model}_model_batchsize{batch_size}_{lr:.1e}constLR_kernelsize{kernel_size}_{args.suffix}.pt')
        torch.save(model.state_dict(), f'/hpcfs/juno/junogpu/miaoyu/bb0n/nEXO_ML/checkpoint_sens/{args.model}/{args.model}_model0_batchsize{batch_size}_{lr:.1e}constLR_kernelsize{kernel_size}_{args.suffix}.pt')

    elif args.mode == 'test':
        if not args.resume:
            print('>>>>> Error: no model to resume !!!')
            return
        else:
            acc = test_accuracy_gcn(model, device, test_loader)
            print(f'Classification accuracy on the test dataset is {acc:.4f}.')
main()
