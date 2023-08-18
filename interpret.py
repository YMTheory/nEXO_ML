import torch 
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


def main():
    network = 'coatnet'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if network == 'resnet18':
        model = resnet18(input_channels=2)
        resumed_model = '/Users/yumiao/Documents/Works/0nbb/nEXO/ML/nEXO_ML/checkpoint_sens/resnet18_best-model-_epoch22_parameters_zepengdata_0811.pt'
        checkpoint = torch.load(resumed_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        
        model.to(device)
        
        h5testfile ='/Users/yumiao/Documents/Works/0nbb/nEXO/ML/dataset/nexo.h5'
        csvtestfile = '/Users/yumiao/Documents/Works/0nbb/nEXO/ML/dataset/nexo_valid.csv'
        test_dataset = StripData(h5testfile, csvtestfile, n_channels=2, process=False)
        print(f'Total test_dataset length is {len(test_dataset)}.')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=0)

    elif network == 'coatnet':
        num_blocks = [2, 2, 6, 14, 2]           # L
        channels = [128, 128, 256, 512, 1026]   # D
        INPUT_SHAPE = [256, 256, 2]
        model = CoAtNet((INPUT_SHAPE[0], INPUT_SHAPE[1]), INPUT_SHAPE[2], num_blocks, channels, num_classes=INPUT_SHAPE[2])        
        resumed_model = '/Users/yumiao/Documents/Works/0nbb/nEXO/ML/nEXO_ML/checkpoint_sens/coatnet_best-model-_epoch28_parameters_zepeng_0813.pt'
        checkpoint = torch.load(resumed_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        
        model.to(device)
        
        h5testfile ='/Users/yumiao/Documents/Works/0nbb/nEXO/ML/dataset/nexo.h5'
        csvtestfile = '/Users/yumiao/Documents/Works/0nbb/nEXO/ML/dataset/nexo_valid.csv'
        test_dataset = StripData(h5testfile, csvtestfile, n_channels=2, process=True)
        print(f'Total test_dataset length is {len(test_dataset)}.')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=0)


    batch = next(iter(test_loader))
    images, targets = batch
    
    if network == 'resnet18':
        target_layers = [model.layer4[-1]]
    elif network == 'coatnet':
        target_layers = [model.s4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    
    targets = None
    #targets = [ClassifierOutputTarget(0)]
    #targets = [BinaryClassifierOutputTarget(1)]    # for bb0n
    
    ## You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=images, targets=targets)

    evtid = 22
    label = test_dataset[evtid][1]
    particle = 'bb0n' if label == 1 else 'gamma'
    print(f"Event {evtid} is a {particle} event.")
    imgx = test_dataset[evtid][0][0]
    imgx[imgx <= 0] = 1
    imgy = test_dataset[evtid][0][1]
    imgy[imgy <= 0] = 1
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    im0 = ax[0].imshow(imgx, aspect='auto', norm=colors.LogNorm(vmin=1))
    ax[0].set_title("x strips")
    cb0 = plt.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(imgy, aspect='auto', norm=colors.LogNorm(vmin=1))
    ax[1].set_title("y strips")
    cb1 = plt.colorbar(im1, ax=ax[1])
    im2 = ax[2].imshow(grayscale_cam[evtid], aspect='auto', cmap='rainbow')
    ax[2].set_title("GradCAM")
    cb2 = plt.colorbar(im2, ax=ax[2])
    
    plt.tight_layout()
    plt.savefig(f"{particle}Event_bb0nclass_event{evtid}.pdf")    
    plt.show()


main()
