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
from net import get_model
import sys







if __name__ == "__main__":
    assert len(sys.argv) == 2 , "Enter the Image path that you want to process"

    model_name = "best_model.pth"
    model = get_model(model_name)
    model = model.to(device)

    img = np.array(Image.open(sys.argv[1]))
    img = np.rollaxis(img,2)
    img = torch.tensor(img).unsqueeze(0).to(device).float()/255
    with torch.no_grad():
        res = model(img)
    res = res.cpu().numpy()[0][0]

    res[res >= .5] = 1
    res[res < .5] =  0

    plt.imshow(res)
    plt.show()
    
    
    
        
