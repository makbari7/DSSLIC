# DSSLIC
Pytorch implmementation of DSSLIC: Deep Semantic Segmentation-based Layered Image Compression

Paper: https://arxiv.org/abs/1806.03348

## Requirements: 
- Ubuntu 16.04
- Python 2.7
- Cuda 8.0
- Pyorch 0.3.0

# Training
## ADE20K Dataset
python train.py --name ADE20K_model --dataroot ./datasets/ADE20K/ --label_nc 151 --loadSize 256 --resize_or_crop resize --batchSize 8
## Cityscapes Dataset
python train.py --name cityscapes_model --dataroot ./datasets/cityscapes/ --label_nc 35 --loadSize 1024 --resize_or_crop scale_width --batchSize 2

# Testing
## ADE20K testset
python test.py --name ADE20K_model --dataroot ./datasets/ADE20K/ --label_nc 151 --resize_or_crop none --batchSize 1 --how_many 50
## Kodak testset
python test.py --name ADE20K_model --dataroot ./datasets/Kodak/ --label_nc 151 --resize_or_crop none --batchSize 1 --how_many 24
## Cityscapes testset
python test.py --name Cityscapes_model --dataroot ./datasets/cityscapes/ --label_nc 35 --loadSize 1024 --resize_or_crop scale_width --batchSize 1 --how_many 50
