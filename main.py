import numpy as np
import cv2
import os
from SynIMU2Blurry import *

SynData = SynImages()
img = cv2.imread("InputImages/001464.png", 1).astype(np.float)
img_H, img_W, _ = img.shape
SynData.image_H = img_H
SynData.image_W = img_W

file_prefix = "001464"
phase = "single"
if not os.path.exists('Dataset') and phase != "single":
    os.makedirs('Dataset')
    os.makedirs('Dataset/train/')
blur_img = SynData.create_syn_images(img, file_prefix, phase, isSave=True, isPlot=False)


