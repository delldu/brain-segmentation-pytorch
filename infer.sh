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

# $ python3 inference.py --help
# usage: inference.py [-h] [--device DEVICE] [--batch-size BATCH_SIZE] --weights
#                     WEIGHTS [--images IMAGES] [--image-size IMAGE_SIZE]
#                     [--predictions PREDICTIONS] [--figure FIGURE]

# Inference for segmentation of brain MRI

# optional arguments:
#   -h, --help            show this help message and exit
#   --device DEVICE       device for training (default: cuda:0)
#   --batch-size BATCH_SIZE
#                         input batch size for training (default: 32)
#   --weights WEIGHTS     path to weights file
#   --images IMAGES       root folder with images
#   --image-size IMAGE_SIZE
#                         target input image size (default: 256)
#   --predictions PREDICTIONS
#                         folder for saving images with prediction outlines
#   --figure FIGURE       filename for DSC distribution figure

python3 inference.py \
  --weights weights/unet.pt \
  --images kaggle_3m \
  --predictions output \
  --figure output/dsc.png


