import argparse
import numpy as np
import time
import os
import yaml
import psutil

from utils.data_loader import StripData
from utils.network_manipulation import train, test

from networks.resnet_example import resnet18, resnet50
from networks.coatnet import CoAtNet
from vit_pytorch import ViT
from vit_pytorch.distill import DistillableViT, DistillWrapper


import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


def yaml_load(config):
    """
    Load configurations from yaml file.
    """
    with open(config) as stream:
        param = yaml.safe_load(stream)
        return param



def main():
    parser = argparse.ArgumentParser(description='PyTorch nEXO PID task.')
    
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

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device for training is {device}. \n")

    if model_name == 'coatnet' or model_name == 'vit':
        full_data = StripData(h5_path, csv_path, INPUT_SHAPE[2], process=True)
    elif model_name == 'resnet18':
        full_data = StripData(h5_path, csv_path, INPUT_SHAPE[2], process=False)
    dataset_size = len(full_data)

    train_size = int(dataset_size * train_split)
    valid_size = dataset_size - train_size

    train_dataset, valid_dataset = random_split(full_data, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f'Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024 : .4f} GB.')
    print(f'batch size = {BATCH_SIZE:d}')
    print(f'Dataset size = {dataset_size:d}')
    print(f"Length of the train_loader {len(train_dataset):d}, length of the valid_loader = {len(valid_dataset):d}.")

    print(f"\n ********** Building model {model_name} ********** ")
    if model_name == 'resnet18':
        model = resnet18(input_channels=INPUT_SHAPE[2])
    elif model_name == "coatnet":
        num_blocks = [2, 2, 6, 14, 2]
        channels = [128, 128, 256, 512, 1026]
        model = CoAtNet((INPUT_SHAPE[0], INPUT_SHAPE[1]), INPUT_SHAPE[2], num_blocks, channels, num_classes=INPUT_SHAPE[2])
    elif model_name == 'vit':
        model = ViT(
                image_size = 256,
                patch_size = 32,
                num_classes = 2,
                channels = 2,
                dim = 1024,
                depth = 6,
                heads = 16,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1
                )
    else:
        raise ValueError('Unknown network architecture.')

    if args.resume:
        resumed_model = '/Users/yumiao/Documents/Works/0nbb/nEXO/ML/nEXO_ML/checkpoint_sens/resnet18_best-model-_epoch22_parameters_zepengdata_0811.pt'
        print(f"Load pre-trained model {resumed_model}.")
        checkpoint = torch.load(resumed_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    print("Loss function: ", loss_func)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY, eps=1e-8, amsgrad=False)
    print('Optimizer name: ', optim_name)

    print (f'\nMemory usage： {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024 :.4f} GB' ) 


    if args.mode == 'train':
        epoch_train_losses, epoch_valid_losses = [], []
        epoch_train_acc, epoch_valid_acc = [], []
        epoch_best = -1
        epoch_best_loss = 1e2

        for epoch in range(EPOCHS):
            start = time.time()
            train_losses, Learning_rate, train_acc  = train(LOG_INTERVAL, model, device, train_loader, optimizer, criterion, epoch, EPOCHS, grad_clip=GRAD_CLIP) # 0719
            valid_losses, valid_acc                 = test(model, device, criterion, valid_loader)

            epoch_train_losses.append(train_losses)
            epoch_valid_losses.append(valid_losses)

            epoch_train_acc.append(train_acc)
            epoch_valid_acc.append(valid_acc)

            end = time.time()
            print('\n Running time: %s min for epoch %d'%((end-start)/60, epoch))    

        A = np.array(epoch_train_losses)
        B = np.array(epoch_valid_losses)
        C = np.array(epoch_train_acc)
        D = np.array(epoch_valid_acc)
        np.save(os.path.join(loss_acc_path, model_name+f"_trainloss_Nepoch{EPOCHS}_batch{BATCH_SIZE}_lr{MAX_LR}_{args.suffix}.npy" ), A)
        np.save(os.path.join(loss_acc_path, model_name+f"_validloss_Nepoch{EPOCHS}_batch{BATCH_SIZE}_lr{MAX_LR}_{args.suffix}.npy" ), B)
        np.save(os.path.join(loss_acc_path, model_name+f"_trainacc_Nepoch{EPOCHS}_batch{BATCH_SIZE}_lr{MAX_LR}_{args.suffix}.npy" ), C)
        np.save(os.path.join(loss_acc_path, model_name+f"_validacc_Nepoch{EPOCHS}_batch{BATCH_SIZE}_lr{MAX_LR}_{args.suffix}.npy" ), D)
        print("SAVED!") 

    elif args.mode == 'test':
        if not args.resume:
            print(">>>>>>>>>>>>>> Error: do not set model path to resume!!! ")
            return
        else:
            if model_name == 'coatnet' or model_name == 'vit':
                test_dataset = StripData(h5_test_path, csv_test_path, n_channels=INPUT_SHAPE[2], process=True)
            else:
                test_dataset = StripData(h5_test_path, csv_test_path, n_channels=INPUT_SHAPE[2], process=False)
            print("Total test data size: ", len(test_dataset))
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)

            _, score = test(model, device, criterion, test_loader)
            np.save(os.path.join(loss_acc_path, model_name+f"_testloss_Nepoch{EPOCHS}_batch{BATCH_SIZE}_lr{MAX_LR}_{args.suffix}.npy" ), score)

    elif args.mode == 'print':
        from torchinfo import summary
        summary(model, (1, INPUT_SHAPE[2], INPUT_SHAPE[0], INPUT_SHAPE[1]))

    else:
        print(f"Error: unknown run mode {args.mode}, options are (train, test, shap).")


main()
