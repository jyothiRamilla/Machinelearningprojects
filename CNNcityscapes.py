# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:39:21 2020

@author: Lenovo
"""

import cv2
import os
import glob
img_dir = "cityscapes_data/train" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
train = []
for f1 in files:
    img = cv2.imread(f1)

    train.append(img)
    
    
for f1 in train:
    print(f1.shape)
    
    
    
for f1 in files:
    img = cv2.imread(f1)
    cv2.imshow('image',img)
    
    
import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('image.jpg',1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()