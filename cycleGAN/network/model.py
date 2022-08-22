import torch
import torch.nn as nn
from torchvision import models

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, 3, bias=False)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 3, bias=False)
        self.relu1_2 = nn.ReLU()
        
        self.conv2_1 = nn.Conv2d(32, 64, 3, bias=False)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 3, bias=False)
        self.relu2_2 = nn.ReLU()
        
        self.conv3_1 = nn.Conv2d(64, 128, 3, bias=False)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 3, bias=False)
        self.relu3_2 = nn.ReLU()
        
        self.conv4_1 = nn.Conv2d(128, 256, 3, bias=False)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(256, 256, 3, bias=False)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.ConvTranspose2d(256, 256, 3, bias=False)
        self.relu4_3 = nn.ReLU()
        
        self.conv5_1 = nn.ConvTranspose2d(256, 128, 3, bias=False)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.ConvTranspose2d(256, 128, 3, bias=False)
        self.relu5_2 = nn.ReLU()
        
        self.conv6_1 = nn.ConvTranspose2d(128, 64, 3, bias=False)
        self.relu6_1 = nn.ReLU()
        self.conv6_2 = nn.ConvTranspose2d(128, 64, 3, bias=False)
        self.relu6_2 = nn.ReLU()
        
        self.conv7_1 = nn.ConvTranspose2d(64, 32, 3, bias=False)
        self.relu7_1 = nn.ReLU()
        self.conv7_2 = nn.ConvTranspose2d(64, 32, 3, bias=False)
        self.relu7_2 = nn.ReLU()
        self.conv7_3 = nn.ConvTranspose2d(32, 1, 3, bias=False)
        self.relu7_3 = nn.ReLU()
        
    def forward(self, input):
#        residue = input
        
        out1_1 = self.conv1_1(input)
        out = self.relu1_1(out1_1)       
        out1_2 = self.conv1_2(out)
        out = self.relu1_2(out1_2)
        
        out2_1 = self.conv2_1(out)
        out = self.relu2_1(out2_1)       
        out2_2 = self.conv2_2(out)
        out = self.relu2_2(out2_2)
        
        out3_1 = self.conv3_1(out)
        out = self.relu3_1(out3_1)       
        out3_2 = self.conv3_2(out)
        out = self.relu3_2(out3_2)
        
        out4_1 = self.conv4_1(out)
        out = self.relu4_1(out4_1)       
        out4_2 = self.conv4_2(out)
        out = self.relu4_2(out4_2)
        out4_3 = self.conv4_3(out)
        out = self.relu4_3(out4_3)
        
        out5_1 = self.conv5_1(out)
        out = torch.cat((out5_1, out3_2), 1)
        out = self.relu5_1(out)       
        out5_2 = self.conv5_2(out)
        out = self.relu5_2(out5_2)
        
        out6_1 = self.conv6_1(out)
        out = torch.cat((out6_1, out2_2), 1)
        out = self.relu6_1(out)       
        out6_2 = self.conv6_2(out)
        out = self.relu6_2(out6_2)
        
        out7_1 = self.conv7_1(out)
        out = torch.cat((out7_1, out1_2), 1)
        out = self.relu7_1(out)       
        out7_2 = self.conv7_2(out)
        out = self.relu7_2(out7_2)
        out7_3 = self.conv7_3(out)
        out = self.relu7_3(out7_3)
        
#        out += residue
        return out 
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.MaxPool2d(4,4),
            
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(4,4),
            
            nn.Conv2d(128, 256, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(4,4),
            )
        self.main  = main
        self.fc1   = nn.Linear(8*8*256, 128)
        self.relu1 = nn.ReLU(True)
        self.fc2   = nn.Linear(128, 1)
        self.relu2 = nn.ReLU(True)

    def forward(self, input):
        out = self.main(input)
        out = out.view(-1, 8*8*256)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        return out   

class Vgg16(torch.nn.Module): # for calculating perceptual loss
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        out = []
        out.append(h_relu1_2)
        out.append(h_relu2_2)
        out.append(h_relu3_3)
        out.append(h_relu4_3)
        return out    