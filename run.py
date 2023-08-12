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
from utils.network_manipulation_parallel import train, validation, test_accuracy, train_lr_sched

# load networks
from networks.resnet_example import resnet18
from networks.preact_resnet import PreActResNet18
from networks.coatnet import CoAtNet
from networks.simple_example import SimpleCNN

#import shap


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
    parser.add_argument('--mode', type=str, default='train', help='Use model for training or test.')
    parser.add_argument('--resume',     dest="resume",   action='store_true',     help='resume from checkpoint')
    parser.add_argument('--no-resume',  dest="resume",   action='store_false',    help='not resume from checkpoint')
    parser.add_argument('--suffix', type=str, default='test', help='suffix for pt and npy files.')

    args = parser.parse_args()

    ###############################################################
    # configure from yaml file
    print(f'The configuration yaml file is {args.config}.')
    config          = yaml_load(args.config)
    h5_path         = config['data']['h5file']
    csv_path        = config['data']['csvfile']
    h5_test_path    = config['data']['h5testfile']
    csv_test_path   = config['data']['csvtestfile']
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

    print("Training h5 file: ",     h5_path)
    print("Training csv file: ",    csv_path)
    print("Test h5 file: ",         h5_test_path)
    print("Test csv file: ",        csv_test_path)
    print("Number of epochs: ",     EPOCHS)
    print("Max learning rate: ",    MAX_LR)
    print("Batch size: ",           BATCH_SIZE)
    print("Number of features: ",   INPUT_SHAPE[2])
    print("Weight decay: ",         WEIGHT_DECAY)
    print("Gradient clip: ",        GRAD_CLIP)
    print("Resume model: ",         args.resume)


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
    if model_name == 'coatnet':
        full_data = StripData(h5_path, csv_path, INPUT_SHAPE[2], process=True)
    else:
        full_data = StripData(h5_path, csv_path, INPUT_SHAPE[2], process=False)
    dataset_size = len(full_data)

    train_size = int(len(full_data)*train_split)  
    test_size = len(full_data) - train_size     

    train_dataset, test_dataset =random_split(full_data, [train_size, test_size])
    #norm = transforms.Normalize([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1])  
    #train_dataset = norm(train_dataset)
    #test_dataset  = norm(test_dataset)

    #train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
    #                          num_workers=n_workers, shuffle=True, pin_memory=True)
    #----- GPU Parallel: (3)-1 DistributedSampler 
    #                    https://github.com/laisimiao/classification-cifar10-pytorch/blob/master/main_ddp.py
    #                BatchSampler  https://www.zhihu.com/search?type=content&q=SyncBatchNorm
    train_sampler = DistributedSampler(train_dataset)  
    test_sampler = DistributedSampler(test_dataset)  
    train_sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler=torch.utils.data.distributed.DistributedSampler(test_dataset)
    train_batch_sampler=torch.utils.data.BatchSampler(train_sampler,BATCH_SIZE,drop_last=True)
    test_batch_sampler=torch.utils.data.BatchSampler(test_sampler,BATCH_SIZE,drop_last=True)
    
    train_loader = DataLoader(train_dataset,  batch_sampler=train_batch_sampler, 
                          num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset,  batch_sampler=test_batch_sampler, 
                          num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)
    #test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,  sampler=test_sampler,
    #                     num_workers=NUM_WORKERS, shuffle=False, pin_memory=True, drop_last=True)
    print(f'memory usage： {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024 :.4f} GB' ) 
    print(f"batch_size = {BATCH_SIZE:d}"  )
    print(f"dataset_size = {dataset_size:d}"  )
    #print("dataset[0][0].size(): ",full_data[0].shape )    
    print("Length of the train_loader:", len(train_loader))
    print("Length of the val_loader:", len(test_loader))
    print("Memory occupied after data loader: ", torch.cuda.max_memory_allocated(device))
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
        net = CoAtNet((INPUT_SHAPE[0], INPUT_SHAPE[1]), INPUT_SHAPE[2], num_blocks, channels, num_classes=INPUT_SHAPE[2])        
        #net = CoAtNet((256, 256), 2, num_blocks, channels, num_classes=Nlabels)
    elif model_name == 'simple':
        net = SimpleCNN()
        #net.relu1.register_forward_hook(get_intermediate_output)
        #net.relu2.register_forward_hook(get_intermediate_output)
        #net.fc1.register_forward_hook(get_intermediate_output)

    else:
        raise ValueError('Unknown network architecture.')
          

    if args.resume:
        if model_name == "resnet18":
            resumed_model = '/hpcfs/juno/junogpu/miaoyu/bb0n/nEXO_ML/checkpoint_sens/resnet18_best-model-parameters_larger_samples_0713_4gpu_noimagescale_epoch10to20_resume.pt'
        elif model_name == "coatnet":
            resumed_model = '/hpcfs/juno/junogpu/miaoyu/bb0n/nEXO_ML/checkpoint_sens/coatnet_best-model-_epoch7_parameters_v0_0721_schedLR_larger_sample_256resize_nonorm_10to20epoch.pt'
            #resumed_model = '/hpcfs/juno/junogpu/miaoyu/bb0n/nEXO_ML/checkpoint_sens/coatnet_best-model-_epoch4_parameters_v0_0719__schedLR_larger_sample_256resize_nonorm_0to10epoch.pt'
            #resumed_model = '/hpcfs/juno/junogpu/miaoyu/bb0n/nEXO_ML/checkpoint_sens/coatnet_best-model-_epoch3_parameters_v0_0719_larger_sample_256resize_nonorm_resumed_16to24epoch.pt'
            #resumed_model = '/hpcfs/juno/junogpu/miaoyu/bb0n/nEXO_ML/checkpoint_sens/coatnet_best-model-_epoch5_parameters_v0_0717_larger_sample_256resize_nonorm_resumed_8to16epoch.pt'
            #resumed_model = '/hpcfs/juno/junogpu/miaoyu/bb0n/nEXO_ML/checkpoint_sens/coatnet_best-model-parameters_v0_0715_larger_sample_256resize_nonorm_resumed_6to13epoch.pt'
            #resumed_model = '/hpcfs/juno/junogpu/miaoyu/bb0n/nEXO_ML/checkpoint_sens/coatnet_best-model-parameters_v0_0713_larger_sample_256resize_nonorm.pt'
        print(f'\n===> Load pre-trained model {resumed_model}.')

        checkpoint = torch.load(resumed_model)
        net.load_state_dict(checkpoint)

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

        #for name, param in net.named_parameters():
        #    print(name, param.shape)

    #'''
    #if args.local_rank == 0:        
    #    summary(net, input_size=(INPUT_SHAPE[2], INPUT_SHAPE[0], INPUT_SHAPE[1]))
    
    if args.mode == "train":
        # define loss function
        criterion = nn.CrossEntropyLoss().cuda()
        print("Loss function: ", loss_func)

        # define optimizer
        optimizer = torch.optim.AdamW(net.parameters(), lr=MAX_LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY, eps=1e-08, amsgrad=False)
        #optimizer = torch.optim.SGD(net.parameters(), lr=MAX_LR, momentum=0.9)
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
        print(f"Memory occupied after model building:  {torch.cuda.max_memory_allocated(device) / 1024/1024/1024: .4f} GB.")

        # 训练前获取初始显存占用情况
        initial_allocated = torch.cuda.memory_allocated()
        initial_cached = torch.cuda.memory_reserved()
        # 打印显存占用情况
        print(f"Initial allocated: {initial_allocated / 1024**3:.2f} GB, Cached: {initial_cached / 1024**3:.2f} GB")

            
        # start training
        for epoch in range(EPOCHS):
            start =time.time()
            #----- GPU Parallel: (3)-2 DistributedSampler 
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

            # 获取显存占用情况
            current_allocated = torch.cuda.memory_allocated()
            current_cached = torch.cuda.memory_reserved()

            print(f"Epoch {epoch}: Allocated: {current_allocated / 1024**3:.2f} GB, Cached: {current_cached / 1024**3:.2f} GB")
            
            # train, test model
            #train_losses, train_acc, Learning_rate      = train(LOG_INTERVAL, net, device, train_loader, optimizer, criterion, epoch, EPOCHS, scheduler, scaler, grad_clip=GRAD_CLIP)
            #test_losses, test_acc                       = validation(net, device, optimizer, criterion, test_loader)
            ## train_losses, Learning_rate      = train(LOG_INTERVAL, net, device, train_loader, optimizer, criterion, epoch, EPOCHS, scaler, grad_clip=GRAD_CLIP) # 0719
            train_losses, Learning_rate      = train_lr_sched(LOG_INTERVAL, net, device, train_loader, optimizer, criterion, epoch, EPOCHS, scheduler, scaler, grad_clip=GRAD_CLIP)  # scheduling learning rate
            test_losses                      = validation(net, device, optimizer, criterion, test_loader)
            
            #scheduler.step()        

            if args.local_rank == 0:        
                if(test_losses<epoch_best_loss) :
                    epoch_best_loss=test_losses
                    epoch_best=epoch
                    #https://discuss.pytorch.org/t/how-to-save-the-best-model/84608 
                    #----- GPU Parallel: (1) local_rank
                    torch.save(net.state_dict(), os.path.join(modelpath, model_name+f"_best-model0-_epoch{epoch}_parameters_{args.suffix}.pt"))  
                    torch.save(net.module.state_dict(), os.path.join(modelpath, model_name+f"_best-model-_epoch{epoch}_parameters_{args.suffix}.pt"))
                # save results
                epoch_Learning_Rates.append(Learning_rate)
        
                epoch_train_losses.append(train_losses)
                epoch_test_losses.append(test_losses)
                #epoch_train_acc.append(train_acc)
                #epoch_test_acc.append(test_acc)

        
            end = time.time()
            print('\n Running time: %s min for epoch %d'%((end-start)/60, epoch))    

        if args.local_rank == 0:             
            print(f'epoch_best_loss = {epoch_best_loss:.6f},  GPU#{args.local_rank}: epoch_best = {epoch_best:.0f}, lr_best = {MAX_LR:.2E}')    
            print("DONE!")    
            # save all train test results
            LR = np.array(epoch_Learning_Rates)    
            A = np.array(epoch_train_losses)
            B = np.array(epoch_test_losses)
            #C = np.array(epoch_train_acc)
            #D = np.array(epoch_test_acc)
            np.save(os.path.join(loss_acc_path, model_name+f"_Nepoch{EPOCHS}_batch{BATCH_SIZE}_LearningRates_{args.suffix}.npy" ), LR)    
            np.save(os.path.join(loss_acc_path, model_name+f"_Ntrain_Nepoch{EPOCHS}_losses_batch{BATCH_SIZE}_{args.suffix}.npy" ), A)
            np.save(os.path.join(loss_acc_path, model_name+f"_Ntest_Nepoch{EPOCHS}_losses_batch{BATCH_SIZE}_{args.suffix}.npy" ), B)
            #np.save(os.path.join(loss_acc_path, model_name+"_N"+str(EPOCHS)+"_Ntrain_acc.npy" ),    C)
            #np.save(os.path.join(loss_acc_path, model_name+"_N"+str(EPOCHS)+"_Ntest_acc.npy" ),     D)
            print("SAVED!") 
    
    elif args.mode == "test":
        if not args.resume:
            print('>>>>> Error: do not set model path to resume !!!')
            return 
        else:
            if model_name == 'coatnet':
                test_dataset = StripData(h5_test_path, csv_test_path, n_channels=INPUT_SHAPE[2], process=True)
            else:
                test_dataset = StripData(h5_test_path, csv_test_path, n_channels=INPUT_SHAPE[2], process=False)
            print("Total test data size: ", len(test_dataset))
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)

            score = test_accuracy(net, device, test_loader)
            np.save(os.path.join(loss_acc_path, model_name+f"_Nepoch{EPOCHS}_batch{BATCH_SIZE}_schedLearningRatesMax{MAX_LR:.1e}_{args.suffix}.npy" ), score)    

            # ---------------------------------------------- #
            #net.eval()

            #total, correct = 0, 0
            #score = []
            #with torch.no_grad():
            #    for batch_idx, (images, labels) in enumerate(test_loader):
            #        images = images.to(device)
            #        labels = labels.to(device)

            #        outputs = net(images)
            #        _, predicted = outputs.max(1)
            #        total += labels.size(0)
            #        correct += predicted.eq(labels).sum().item()

            #        softmax = nn.Softmax(dim=0)
            #        for m in range(outputs.size(0)):
            #            score.append([softmax(outputs[m])[1].item(), labels[m].item()])

            #        print(f"Batch [{batch_idx+1} / {len(test_loader)}] has average accuracy: {correct/total:.2f}.")
            #    acc = 1. * correct / total 

            #    print(f"Accuracy on the test dataset is {acc:.2f}")
        
            #np.save(os.path.join(loss_acc_path, model_name+f'test_score_1e-6LR_resumed_totalEpoch60.npy'), score)
            # ---------------------------------------------- #

    #elif args.mode == 'shap':
    #    if not args.resume:
    #        print('>>>>> Error: do not set model path to resume !!!')
    #        return 
    #    else:
    #        test_dataset = StripData(h5_test_path, csv_test_path, n_channels=INPUT_SHAPE[2])
    #        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=1)

    #        batch = next(iter(test_loader))
    #        images, _ = batch
    #        images = images.view(-1, 2, 250, 250)
    #        background = images[:100]
    #        test_images= images[100:110]

    #        e = shap.DeepExplainer(net, images)
    #        shap_values = e.shap_values(test_images)

    #        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    #        test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

    #        shap.image_plot(shap_numpy, -test_numpy)

    else:
        print(f"Error: unknown run mode {args.mode}, options are (train, test, shap).")

main()
