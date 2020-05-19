import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
import torch
from torchvision.transforms import functional as tvf
import torchvision
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random
import scipy.ndimage.filters
import multiprocessing
from multiprocessing import Process
from multiprocessing import Queue 
import time


paths = {
"images_path" : "./JPEGImages",
"targets_path" : "./SegmentationObjectFilledDenseCRF",
"train_names_path" : "./train.txt",
"val_names_path" : "./val.txt",
"new_contours_path": "./improved_contours"
}

for p in paths.values():
    if(os.path.exists(p) == False):
        print("path " , p , "does not exist")


def Filter_util(arr , i , j):
    val = arr[i][j]
    if(i - 1 >= 0):
        if(arr[i-1][j] != val):
            return True
        if(j -1 >= 0):
            if(arr[i-1][j-1] != val):
                return True
        if(j + 1 < arr.shape[1]):
            if(arr[i-1][j+ 1] != val):
                return True

    if(i + 1 < arr.shape[0]):
        if(arr[i+1][j] != val):
            return True
        if(j -1 >= 0):
            if(arr[i+1][j-1] != val):
                return True
        if(j + 1 < arr.shape[1]):
            if(arr[i+1][j+ 1] != val):
                return True
            
    if(j - 1 >= 0):
        if(arr[i][j-1] != val):
            return True
    if(j + 1 < arr.shape[1]):
        if(arr[i][j+1] != val):
            return True
    return False

def Filter(img):
    filter_size = 3
    pad = filter_size//2
    new_img = np.zeros_like(img)
    for i in range(pad , img.shape[0] - pad):
        for j in range(pad, img.shape[1] - pad):
            pixel_val = img[i][j]
            if(Filter_util(img,i,j)):
                new_img[i][j] = 1

        
    return new_img
            


names = os.listdir(paths["targets_path"])

for name in names:
    path = os.path.join(paths["targets_path"], name)
    img = np.array(Image.open(path))
    img = Filter(img)
    save_path = os.path.join(paths["new_contours_path"], name)
    plt.imsave(save_path,img, cmap = "gray")


