import numpy as np
import cv2
import os
from SynIMU2Blurry import *

if not os.path.exists('Dataset'):
    os.makedirs('Dataset')
    os.makedirs('Dataset/train/Data_ref')
    os.makedirs('Dataset/train/Data_ori')
    os.makedirs('Dataset/train/Data_err')

SynData = SynImages()
img = cv2.imread("InputImages/001464.png", 1).astype(np.float)
#img = cv2.imread("InputImages/checkerboard2.png", 1).astype(np.float)
img_H, img_W, _ = img.shape
SynData.image_H = img_H
SynData.image_W = img_W

file_prefix = "001464" #""001464"
blur_img = SynData.create_syn_images(img, file_prefix, "train", isSave=True, isPlot=False)
#cv2.imwrite("Output/blurry_001464.png", blur_img)

