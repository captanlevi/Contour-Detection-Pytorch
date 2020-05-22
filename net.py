import numpy as np
import torch
from torchvision.transforms import functional as tvf
import torchvision
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os


paths = {
"images_path" : "./JPEGImages",
"targets_path" : "./improved_contours",
"train_names_path" : "./train.txt",
"val_names_path" : "./dummy_val.txt",
"results_path" : "./results",
"models_path" : "./models",
"CRF_mask_path" :"./SegmentationObjectFilledDenseCRF"
}



vgg16 =  torchvision.models.vgg16(pretrained= True)


# Extraction till the first fully connected layer of vgg16 
class Reshape(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self ,x):
        return x.view(x.size(0),-1)


vgg = list(vgg16.children())[0]
"""list(vgg16.children())[1] ,"""
vgg = nn.Sequential(*vgg,  Reshape() ,list(vgg16.children())[2][0] )
for p in vgg.parameters():
    p.requires_grad = False




class Encoder(nn.Module):
    def __init__(self, vgg):
        super().__init__()
        self.vgg = list(vgg.children())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv6 = nn.Conv2d(in_channels = 512, out_channels = 4096, kernel_size = 3, stride=1, padding = 1)
    def forward(self,x):

        pooling_info = {}
        layer_info = {}
        # Starting conv1
        x = self.vgg[0](x)
 
        x = self.vgg[1](x)
        x = self.vgg[2](x)
        x = self.vgg[3](x)
        shape = x.shape
        
        layer_info[1] = {"value": x}
        x , ind = self.pool1(x)
        pooling_info[1] = {"kernel_size" : 2, "stride": 2, "padding": 0 ,"output_size": shape,"indices":ind}
        

        # start conv2
        x = self.vgg[5](x)
        x = self.vgg[6](x)
        x = self.vgg[7](x)
        x = self.vgg[8](x)

        shape = x.shape
        layer_info[2] = {"value": x}
        x , ind = self.pool2(x)
        pooling_info[2] = {"kernel_size" : 2, "stride": 2, "padding": 0 ,"output_size": shape,"indices":ind}



        # start conv3
        x = self.vgg[10](x)
        x = self.vgg[11](x)
        x = self.vgg[12](x)
        x = self.vgg[13](x)
        x = self.vgg[14](x)
        x = self.vgg[15](x)

        shape = x.shape
        layer_info[3] = {"value": x}
        x , ind = self.pool3(x)
        pooling_info[3] = {"kernel_size" : 2, "stride": 2, "padding": 0 ,"output_size": shape,"indices":ind}
  

        x = self.vgg[17](x)
        x = self.vgg[18](x)
        x = self.vgg[19](x)
        x = self.vgg[20](x)
        x = self.vgg[21](x)
        x = self.vgg[22](x)


        shape = x.shape
        layer_info[4] = {"value": x}
        x , ind = self.pool4(x)
        pooling_info[4] = {"kernel_size" : 2, "stride": 2, "padding": 0 ,"output_size": shape,"indices":ind}
      


        x = self.vgg[24](x)
        x = self.vgg[25](x)
        x = self.vgg[26](x)
        x = self.vgg[27](x)
        x = self.vgg[28](x)
        x = self.vgg[29](x)

        shape = x.shape
        layer_info[5] = {"value": x}
        x , ind = self.pool5(x)
        pooling_info[5] = {"kernel_size" : 2, "stride": 2, "padding": 0 ,"output_size": shape,"indices":ind}
    
        x = self.conv6(x)
      #  x = self.vgg[31](x)
      #  x = self.vgg[32](x)
        
        return x , pooling_info, layer_info
         

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.dconv6 = nn.Conv2d(in_channels = 4096, out_channels = 512, kernel_size = 1, stride=1)
       
        self.deconv5 = nn.ConvTranspose2d(in_channels = 512, out_channels = 512, kernel_size = 5, padding =2)
        self.deconv4 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 5 , padding = 2)
        self.deconv3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 5 ,padding = 2)
        self.deconv2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 5 , padding = 2)
        self.deconv1 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 5 ,padding = 2)
        self.pred = nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 5, padding = 2)

    def forward(self,encoder_out):
        x = encoder_out[0]
        dicts = encoder_out[1]


        x = self.dconv6(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_unpool2d(x, **dicts[5])

        
     
        x = self.deconv5(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_unpool2d(x, **dicts[4])  # Indices 512


        x = self.deconv4(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_unpool2d(x, **dicts[3])  # Indices 256


        x = self.deconv3(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_unpool2d(x, **dicts[2])  # Indices 128


        x = self.deconv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_unpool2d(x, **dicts[1])  # Indices 64
        x = self.deconv1(x)
        x = nn.functional.relu(x)

        x = self.pred(x)

        x = torch.sigmoid(x)
        return x




class countour_detector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = Encoder(backbone.to(device)).to(device)
        self.decoder = Decoder().to(device)

    def forward(self,x):
        x = self.encoder(x)
        return self.decoder(x)


def get_model(model_name = None):
    model = countour_detector(vgg)
    if(model_name == None):
        return model
    model.load_state_dict(torch.load(os.path.join(paths["models_path"],model_name)))
    return model
    
