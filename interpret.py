import torch 
import torchvision

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from utils.data_loader import StripData
from networks.resnet_example import resnet18
from networks.coatnet import CoAtNet

import warnings
warnings.filterwarnings("ignore")

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)

sys.path.append('/Users/yumiao/Documents/Works/github/Transformer-Explainability')
from baselines.ViT.ViT_explanation_generator import LRP

def feature_map_vis(inp, figname):

    fig, ax = plt.subplots(figsize=(8, 6))
    im0 = ax.imshow(inp, aspect='auto')
    cb0 = plt.colorbar(im0, ax=ax)

    #plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig(figname)
    plt.show()


def main():
    network = 'resnet18'
    if len(sys.argv) > 1:
        network = sys.argv[1]
    if len(sys.argv) > 2:
        EVTID = int(sys.argv[2])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 1
    
    if network == 'resnet18':
        model = resnet18(input_channels=2)
        resumed_model = '/Users/yumiao/Documents/Works/0nbb/nEXO/ML/nEXO_ML/checkpoint_sens/resnet18_best-model-_epoch22_parameters_zepengdata_0811.pt'
        print(f'Resume network -- {resumed_model}.')
        checkpoint = torch.load(resumed_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        
        model.to(device)
        
        h5testfile ='/Users/yumiao/Documents/Works/0nbb/nEXO/ML/dataset/nexo.h5'
        csvtestfile = '/Users/yumiao/Documents/Works/0nbb/nEXO/ML/dataset/nexo_valid.csv'
        test_dataset = StripData(h5testfile, csvtestfile, n_channels=2, process=False)
        print(f'Total test_dataset length is {len(test_dataset)}.')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)

    elif network == 'coatnet':
        num_blocks = [2, 2, 6, 14, 2]           # L
        channels = [128, 128, 256, 512, 1026]   # D
        INPUT_SHAPE = [256, 256, 2]
        model = CoAtNet((INPUT_SHAPE[0], INPUT_SHAPE[1]), INPUT_SHAPE[2], num_blocks, channels, num_classes=INPUT_SHAPE[2])        
        resumed_model = '/Users/yumiao/Documents/Works/0nbb/nEXO/ML/nEXO_ML/checkpoint_sens/coatnet_best-model-_epoch28_parameters_zepeng_0813.pt'
        print(f'Resume network -- {resumed_model}.')
        checkpoint = torch.load(resumed_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        
        model.to(device)
        
        h5testfile ='/Users/yumiao/Documents/Works/0nbb/nEXO/ML/dataset/nexo.h5'
        csvtestfile = '/Users/yumiao/Documents/Works/0nbb/nEXO/ML/dataset/nexo_valid.csv'
        test_dataset = StripData(h5testfile, csvtestfile, n_channels=2, process=True)
        print(f'Total test_dataset length is {len(test_dataset)}.')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)


    #batch = next(iter(test_loader))
    for i, (images, labels) in enumerate(test_loader):
        if i == EVTID:
            break

    print("Images shape: ", images.shape)
    print("Truth labels for the current event: ", labels)
    
    outputs = model(images)
    ###feature_map = model.featuremap1.transpose(1, 0).cpu()
    ###print(feature_map.shape)

    ####out = torchvision.utils.make_grid(feature_map, padding=1)
    ###img_per_row = 16
    ###tot_col = int(img_per_row*feature_map.shape[3])
    ###tot_row = int(feature_map.shape[0] / img_per_row * feature_map.shape[2])
    ###print(tot_row, tot_col)
    ###feature_map_reshape = np.zeros((tot_row, tot_col))
    ###
    ###for i in range(feature_map.shape[0]):
    ###    colid0 = i % img_per_row * feature_map.shape[3]
    ###    for j in range(feature_map.shape[3]):
    ###        colid = colid0 + j
    ###        for k in range(feature_map.shape[2]):
    ###            rowid0 = i // img_per_row * feature_map.shape[2]
    ###            rowid = rowid0 + k
    ###            feature_map_reshape[rowid, colid] = feature_map[i, :, k, j]
    ###
    ###print(feature_map[119, :, :, :])

    ###fig, ax = plt.subplots()
    ###ax.imshow(feature_map[11, 0, :, :], aspect='auto')
    ###plt.show()

    #figurename = f'../../figures/feature_map_layer4_{network}_evtid{EVTID}.jpg'
    #feature_map_vis(feature_map_reshape, figurename)


    ## GradCAM toolkit for visualization of networks
    if network == 'resnet18':
        target_layers = [model.layer4[-1]]

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        
        #targets = None
        targets = [ClassifierOutputTarget(1) for i in range(BATCH_SIZE)]
        #targets = [BinaryClassifierOutputTarget(1)]    # for bb0n
        
        ## You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=images, targets=targets, aug_smooth=True)

        gb_model = GuidedBackpropReLUModel(model, use_cuda=False)
        gb = gb_model(images, target_category=None)

        cam_mask = np.zeros((images.shape[2], images.shape[3], 2))
        cam_mask[:, :, 0] = grayscale_cam[0]
        cam_mask[:, :, 1] = grayscale_cam[0]
        #gb_cam = deprocess_image(cam_mask * gb)
        #gb = deprocess_image(gb)
        gb_cam = cam_mask * gb

        particle = 'bb0n' if labels == 1 else 'gamma'
        input_path = f'input_{particle}Event_{network}_EVTID{EVTID}.jpg'
        cam_path = f'cam_{particle}Event_{network}_EVTID{EVTID}.jpg'
        gd_path = f'gd_{particle}Event_{network}_EVTID{EVTID}.jpg'

        imgx = test_dataset[EVTID][0][0]
        imgx[imgx <= 0] = 1
        imgy = test_dataset[EVTID][0][1]
        imgy[imgy <= 0] = 1
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        im0 = ax[0].imshow(imgx, aspect='auto', norm=colors.LogNorm(vmin=1))
        ax[0].set_title("x strips")
        cb0 = plt.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(imgy, aspect='auto', norm=colors.LogNorm(vmin=1))
        ax[1].set_title("y strips")
        cb1 = plt.colorbar(im1, ax=ax[1])
        plt.tight_layout()
        plt.savefig(f"../../figures/{input_path}")  

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        im0 = ax[0].imshow(gb[:, :, 0], aspect='auto',)
        ax[0].set_title("x strips")
        cb0 = plt.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(gb[:, :, 1], aspect='auto', )
        ax[1].set_title("y strips")
        cb1 = plt.colorbar(im1, ax=ax[1])
        plt.tight_layout()
        plt.savefig(f"../../figures/{gd_path}")  

        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        im0 = ax[0].imshow(gb_cam[:, :, 0], aspect='auto',)
        ax[0].set_title("guided gradCAM")
        cb0 = plt.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(gb_cam[:, :, 1], aspect='auto',)
        ax[1].set_title("guided gradCAM")
        cb1 = plt.colorbar(im1, ax=ax[1])
        im2 = ax[2].imshow(grayscale_cam[0], aspect='auto',)
        ax[2].set_title("gradCAM")
        cb2 = plt.colorbar(im2, ax=ax[2])
        plt.tight_layout()
        plt.savefig(f"../../figures/{cam_path}")  


    elif network == 'coatnet':

main()
