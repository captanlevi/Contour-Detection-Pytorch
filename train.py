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
import torch.nn.functional as F
import sys
from net import get_model

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



class dataloader:
    def __init__(self,paths):
        self.paths = paths
        self.train_images_pil = []
        self.train_tar_pil = []
        self.make_train_val_names()
        self.pointer = 0
        self.index_arr = [x for x in range(len(self.train_names))]

        
    def make_train_val_names(self):
        with open(paths["train_names_path"]) as handle:
            orignal_train_names = [x.split("\n")[0].strip() for x in handle]
        with open(paths["val_names_path"]) as handle:
            val_names = [x.split("\n")[0].strip() for x in handle]
        self.train_names = []
        classaug_imgs = os.listdir(self.paths["targets_path"])
        val_set =set(val_names)
        for name in classaug_imgs:
            name = name.split(".")[0]
            if name not in val_set:
                self.train_names.append(name)
        self.train_names += orignal_train_names

        self.train_names = list(set(self.train_names))
        self.val_names = list(val_set)


    def __len__(self):
        return len(self.train_names)

    def reset_loader(self):
        random.shuffle(self.index_arr)
        self.pointer = 0

    def transform(self, image_origin, mask_origin, mode, data_augmentation = "randomcrop"):
        image_res, mask_res = None, None
        totensor_op = transforms.ToTensor()
        color_op = transforms.ColorJitter(0.1, 0.1, 0.1)
        resize_op = transforms.Resize((224, 224))
        image_origin = color_op(image_origin)
      #  norm_op = transforms.Normalize(mean =[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

        if mode == 'val' or mode == 'predict':
            image_res = totensor_op(image_origin)
            mask_res = totensor_op(mask_origin)
        elif mode == 'train':
            if data_augmentation == 'randomcrop':
                if image_origin.size[0] < 224 or image_origin.size[1] < 224:
                    #padding-val:
                    val = int(np.array(image_origin).sum() / image_origin.size[0] / image_origin.size[1])
                    padding_width = 224-min(image_origin.size[0],image_origin.size[1])
                    padding_op = transforms.Pad(padding_width,fill=val)
                    image_origin = padding_op(image_origin)
                    padding_op = transforms.Pad(padding_width, fill=0)
                    mask_origin = padding_op(mask_origin)
                i, j, h, w = transforms.RandomCrop.get_params(
                    image_origin, output_size=(224, 224)
                )
                image_res = totensor_op(tvf.crop(image_origin, i, j, h, w))
                mask_res = totensor_op(tvf.crop(mask_origin, i, j, h, w))

            elif data_augmentation == 'resize':
                image_res = totensor_op(resize_op(image_origin))
                mask_res = totensor_op(resize_op(mask_origin))
      #  image_res = norm_op(image_res)
     
        return image_res, mask_res


    def _make_dicts(self):
        count= 0
        for img_name in self.train_names:
            count += 1
            print(count)
            img_path = os.path.join(self.paths["images_path"] , img_name + ".jpg")
            mask_path = os.path.join(self.paths["targets_path"] , img_name + ".png")
            img = Image.open(img_path)
            tar = Image.open(mask_path)
            self.train_images_pil.append(img)
            self.train_tar_pil.append(tar)
        
    
    
    def get_next_mini_batch(self, index_given= None):
            if(self.pointer == len(self.train_names)):
                self.reset_loader()
            if(index_given == None):
                index = self.pointer
            else:
                index = index_given
            self.pointer += 1
            img_name = self.train_names[index]
            img_path = os.path.join(self.paths["images_path"] , img_name + ".jpg")
            mask_path = os.path.join(self.paths["targets_path"] , img_name + ".png")
            img = Image.open(img_path)
            tar = cv2.imread(mask_path,0)
            tar = Image.fromarray(tar)
            img_batch = torch.empty(8,3,224,224)
            tar_batch = torch.empty(8,1,224,224)

            for i in range(4):
                img_batch[i] , tar_batch[i] = self.transform(img,tar,"train")
            img = tvf.hflip(img)
            tar = tvf.hflip(tar)

            for i in range(4,8):
                img_batch[i] , tar_batch[i] = self.transform(img,tar,"train")

           # tar_batch[tar_batch > .9] = 1
           # tar_batch[tar_batch <= .9] = 0
            return img_batch , tar_batch
            
            
    def get_next_batch(self, batch_size):
        index = np.random.randint(0, len(self.train_names), size = batch_size)
        img_batch = torch.empty(batch_size,3,224,224)
        tar_batch = torch.empty(batch_size,1,224,224)

        for i in range(batch_size):
            img_name = self.train_names[index[i]]
            img_path = os.path.join(self.paths["images_path"] , img_name + ".jpg")
            mask_path = os.path.join(self.paths["targets_path"] , img_name + ".png")
            img = Image.open(img_path)
            tar = cv2.imread(mask_path,0)
            tar = Image.fromarray(tar)
            
            if(random.random() > .5):
                img = tvf.hflip(img)
                tar = tvf.hflip(tar)
            img_batch[i] , tar_batch[i] =  self.transform(img,tar,"train")

        return img_batch, tar_batch

    


class trainer:
    def __init__(self, model):
        self.model = model.to(device)
        self.model.train()
        self.lr = 1e-4
        self.optimizer = torch.optim.Adam([x for x in list(self.model.parameters()) if x.requires_grad == True], lr=self.lr)
        self.bce =  nn.BCELoss(reduction = "none")
        self.mse = nn.MSELoss(reduction = "none")
        self.loss_array = []


    def loss(self,outputs, targets):
        weights = torch.empty_like(targets).to(device)
        weights[targets >= .98] = 10
        weights[targets < .98] = 1
        loss = F.binary_cross_entropy(outputs, targets, weights)



        return loss 

    def train(self, data):
            total_loss = 0
            images, targets = data[0], data[1]
            images = images.to(device)
            targets = targets.to(device)
            pred = self.model(images)
            loss = self.loss(pred , targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("loss = ",loss.item())

    def save_model(self , name):
         torch.save(self.model.cpu().state_dict(), os.path.join(paths["model_save_path"] , "model" + name + ".pth"))
         self.model.to(device)






Q = Queue()
def producer(times, dataobject, Q, batch_size = None):
    while(times > 0):
        while(Q.qsize() > 100):
            time.sleep(5)
            continue
        index = random.randint(0, len(dataobject) - 1)
        Q.put(dataobject.get_next_mini_batch(index))
        times -= 1
    Q.put("End")
    time.sleep(50)


class multiprocess_control:
    def __init__(self, dataobject, workers, produce_func, shared_Q, epochs_desired, train_object, offset):
        self.dataobject = dataobject
        self.workers = workers
        self.produce_func = produce_func
        self.shared_Q = shared_Q
        self.desired_epochs = epochs_desired
        self.train_object = train_object
        self.offset = offset
    def spawn_producers(self, batch_size= None):
        times = len(self.dataobject)*self.desired_epochs
        if(times < self.workers):
            print("too many workers")
            return None
        produce_arr =  []
        
        for i in range(self.workers):
            if(i == self.workers - 1):
                reps = times//self.workers + self.desired_epochs%self.workers
            else:
                reps = times//self.workers
            produce_arr.append(Process(target = self.produce_func, args = (reps  ,self.dataobject,self.shared_Q,batch_size,)))
        
        return produce_arr

    
    def start_training(self, batch_size= None):
        arr = self.spawn_producers(batch_size)
        for p in arr:
            p.start()
        time.sleep(1)
        end_count = 0
        iters = 0
        offset = self.offset
        while(end_count < self.workers):
            if(self.shared_Q.empty()):
                time.sleep(5)
                continue
            else:
                data = Q.get()
                if(data == "End"):
                    print("Hello")
                    end_count += 1
                    continue
                else:
                    iters += 1
                    if(iters%len(self.dataobject) == 0):
                        print("Saving Model")
                        self.train_object.save_model(str(iters//len(self.dataobject) + offset))
                       
                    print(iters, end = " ")
                    self.train_object.train(data)
        

        for p in arr:
            p.join()




if __name__ == "__main__":
    assert len(sys.argv) <= 2 , "Enter the the model number to resume from"
    last_model = 0
    if(len(sys.argv) == 2):
        last_model  = sys.argv[1]
        model = get_model("model" + str(last_model) + ".pth")
    else:
        model = get_model()

    data_object = dataloader(paths)
    
    T = trainer(model)



    multi = multiprocess_control(data_object,workers = 2,produce_func = producer, shared_Q = Q,epochs_desired = 30,train_object =T, offset = last_model)
    multi.start_training()























