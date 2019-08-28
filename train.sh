#/************************************************************************************
#***
#***	Copyright 2019 Dell(18588220928@163.com), All Rights Reserved.
#***
#***	File Author: Dell, 2019-08-28 09:05:49
#***
#************************************************************************************/
#
#!/bin/bash
#

# $ python3 train.py --help
# usage: train.py [-h] [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--lr LR]
#                 [--device DEVICE] [--workers WORKERS]
#                 [--vis-images VIS_IMAGES] [--vis-freq VIS_FREQ]
#                 [--weights WEIGHTS] [--logs LOGS] [--images IMAGES]
#                 [--image-size IMAGE_SIZE] [--aug-scale AUG_SCALE]
#                 [--aug-angle AUG_ANGLE]

# Training U-Net model for segmentation of brain MRI

# optional arguments:
#   -h, --help            show this help message and exit
#   --batch-size BATCH_SIZE
#                         input batch size for training (default: 16)
#   --epochs EPOCHS       number of epochs to train (default: 100)
#   --lr LR               initial learning rate (default: 0.001)
#   --device DEVICE       device for training (default: cuda:0)
#   --workers WORKERS     number of workers for data loading (default: 4)
#   --vis-images VIS_IMAGES
#                         number of visualization images to save in log file
#                         (default: 200)
#   --vis-freq VIS_FREQ   frequency of saving images to log file (default: 10)
#   --weights WEIGHTS     folder to save weights
#   --logs LOGS           folder to save logs
#   --images IMAGES       root folder with images
#   --image-size IMAGE_SIZE
#                         target input image size (default: 256)
#   --aug-scale AUG_SCALE
#                         scale factor range for augmentation (default: 0.05)
#   --aug-angle AUG_ANGLE
#                         rotation angle range in degrees for augmentation
#                         (default: 15)


python3 train.py \
	--images kaggle_3m \
	--batch-size 4 \
	--workers 2 \
	--weights output \
	--logs output
