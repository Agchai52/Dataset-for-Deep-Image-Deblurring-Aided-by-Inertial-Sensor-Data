import numpy as np
import cv2
from SynIMU2Blurry import *


img = cv2.imread("InputImages/001464.png", 1).astype(np.float)

SynData = SynImages()
blur_img, time_stamp, gyro, acc = SynData.create_syn_images(img, isPlot=True)
cv2.imwrite("Output/blurry_001464.png", blur_img)