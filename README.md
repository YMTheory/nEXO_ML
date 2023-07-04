# PID task in nEXO
`Author: Miao Yu`

This package is used for DL-based PID/reconstruction... tasks in nEXO experiment.

## Dataset:

MC files from nEXO offline simulation, converted to 2D images (utils/data_manipulation.py) and saved into npy -> h5 files (utils/convertNPYtoH5.py, utils/h5merger.py).

## Network

ResNet network: networks/resnet_example.py (CNN)

CoAtNet network: networks/coatnet.py (CNN + Transformer)

## Configuration
All configuration paramters like filename, model parameters... are configured by reading yaml files.


