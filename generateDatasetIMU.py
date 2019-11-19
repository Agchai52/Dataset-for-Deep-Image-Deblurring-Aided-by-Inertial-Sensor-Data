import numpy as np
import cv2
import os
from SynIMU2Blurry import *
import argparse
from glob import glob

parser = argparse.ArgumentParser('Geneate Dataset IMU Blurry')
parser.add_argument('--output_fold', dest='output_fold', help='output directory', type=str, default='./Dataset/')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=110000000)
parser.add_argument('--phase', dest='phase', help='test or train', type=str, default='train')
args = parser.parse_args()

# Input and Output Folders
img_input_fold = './' + args.phase
img_output_fold = args.output_fold + args.phase + '/'


if not os.path.isdir(img_output_fold):
    os.makedirs(img_output_fold)

# Set jump of images
N = 5

# folders in /test or /train
splits = os.listdir(img_input_fold)

counter = 0
count_sets = 0
SynData = SynImages()

for folder in splits:
    print(folder)
    if not folder.startswith('.'):  # and folder == 'GOPR0385_11_01':
        img_subset = os.path.join(img_input_fold, folder)
        print(img_subset)
        img_list = os.listdir(img_subset)
        img_list = sorted(img_list, key=str.lower)
        num_imgs = min(args.num_imgs, len(img_list))

        out_num, extra_num = divmod(num_imgs, N)
        counter += 1
        prefix = '%02d' % counter

        for n in range(0, out_num):
            name_img = img_list[n*N]
            file_prefix = name_img[:-4]
            path_in = os.path.join(img_subset, name_img)
            img = cv2.imread(path_in, cv2.IMREAD_COLOR).astype(np.float)
            H, W, _ = img.shape
            SynData.image_H = H
            SynData.image_W = W
            # img = ((img / 255) ** 2.2) * 255  # gamma correction, now is linear
            # img_blur = img_blur / 255 ** (1./ 2.2) * 255
            img_blur = SynData.create_syn_images(img, file_prefix, args.phase, isSave=True, isPlot=False)
            count_sets += 1

print("Done!")
print("The total number of sets = ", count_sets)
