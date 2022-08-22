import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

#%%
class SpatialAttention3D(nn.Module):
    """Spatial Attention Layer"""
    def __init__(self):
        super(SpatialAttention3D, self).__init__()

    def forward(self, x0):
        # global cross-channel averaging # e.g. b,512,8,8,8
        x = x0.mean(1, keepdim=True)  # e.g. b,1,8,8,8
        d = x.size(2)
        h = x.size(3)
        w = x.size(4)
        x = x.view(x.size(0),-1)     # e.g. b,8*8*8
        z = x
        for b in range(x.size(0)):
            z[b] /= torch.sum(z[b])
        z = z.view(x.size(0),1,d,h,w)
        return z * x0
    

class ChannelAttention3D(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention3D, self).__init__()
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(num_features, num_features // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(num_features // reduction, num_features, kernel_size=1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x0 = self.pool(x)
        x0 = self.conv1(x0)
        x0 = self.relu(x0)
        x0 = self.conv2(x0)
        x0 = self.sigm(x0)
        return x * x0


class StdConv3d(nn.Conv3d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3, 4], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv3d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv3d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv3d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(16, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(16, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(16, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

#%%
class FeatureExtraction(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self):
        super(FeatureExtraction, self).__init__()
        
        super().__init__()
        width = 32
        # block_units=(2,2,2,2,2,2)
        self.width = width
        
        self.first_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(2, width, kernel_size=3, stride=1, bias=False, padding=1)),
            ('gn', nn.GroupNorm(16, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))  # b,32,64,256,256
        
        self.block1 = nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width+1, cout=width*1, cmid=width*1, stride=2))]
                # [(f'unit{i:d}', PreActBottleneck(cin=width*1, cout=width*1, cmid=width*1)) for i in range(2, block_units[0])] +
                # [(f'unit{i:d}', PreActBottleneck(cin=width*1, cout=width*2, cmid=width*1)) for i in range(block_units[0], block_units[0]+1)],
                )) # b,32,32,128,128
            
        self.block2 = nn.Sequential(OrderedDict(            
                [('unit1', PreActBottleneck(cin=width*1+1, cout=width*2, cmid=width*1, stride=(1,2,2)))]
                # [(f'unit{i:d}', PreActBottleneck(cin=width*2, cout=width*2, cmid=width*2)) for i in range(2, block_units[1])] +
                # [(f'unit{i:d}', PreActBottleneck(cin=width*2, cout=width*4, cmid=width*2)) for i in range(block_units[1], block_units[1]+1)],
                )) # b,64,32,64,64
        
        self.block3 = nn.Sequential(OrderedDict(            
                [('unit1', PreActBottleneck(cin=width*2+1, cout=width*2, cmid=width*2, stride=2))]
                # [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width*4)) for i in range(2, block_units[2])] +
                # [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*4)) for i in range(block_units[2], block_units[2]+1)],
                )) # b,64,16,32,32
        
        self.block4 = nn.Sequential(OrderedDict(            
                [('unit1', PreActBottleneck(cin=width*2+1, cout=width*4, cmid=width*2, stride=(1,2,2)))]
                # [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*8)) for i in range(2, block_units[3])] +
                # [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=320, cmid=width*8)) for i in range(block_units[3], block_units[3]+1)],
                )) # b,128,16,16,16
        
        self.block5 = nn.Sequential(OrderedDict(            
                [('unit1', PreActBottleneck(cin=width*4+1, cout=width*4, cmid=width*4, stride=2))]
                # [(f'unit{i:d}', PreActBottleneck(cin=320, cout=320, cmid=320)) for i in range(2, block_units[4])] +
                # [(f'unit{i:d}', PreActBottleneck(cin=320, cout=320, cmid=320)) for i in range(block_units[4], block_units[4]+1)],
                )) # b,128,8,8,8
        
        self.maxpool1 = nn.MaxPool3d(3,stride=(1,2,2),padding=1)
        self.maxpool2 = nn.MaxPool3d(3,stride=2,padding=1)       
        
    def forward(self, x, prob):
        b, c, _, _, _ = x.size()
        # print("input size:",x.size())
        x = torch.cat([x, prob], dim=1)
        x = self.first_conv(x)
        # prob = self.maxpool1(prob)
        
        x = torch.cat([x, prob], dim=1)        
        x = self.block1(x)
        prob = self.maxpool2(prob)
        
        x = torch.cat([x, prob], dim=1)
        x = self.block2(x)
        prob = self.maxpool1(prob)
        
        x = torch.cat([x, prob], dim=1)
        x = self.block3(x)
        prob = self.maxpool2(prob)

        x = torch.cat([x, prob], dim=1)
        x = self.block4(x)
        prob = self.maxpool1(prob)

        x = torch.cat([x, prob], dim=1)
        x = self.block5(x)

        return x

#%%
class ClassificationNetwork(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, num_classes=2):
        super().__init__()
        
        self.t1_net = FeatureExtraction() # T1CE feature extraction subnetwork
        self.t2f_net = FeatureExtraction()  # FSPGR feature extraction subnetwork    
        self.ca = ChannelAttention3D(256, 16) # channel attention module
        self.sa = SpatialAttention3D() # spatial attention module
       
        self.fc1 = nn.Linear(256*8*8*8, 128) # fully-connected layers for classification
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        
    def forward(self, t1, t2f, prob):
        b, c, d, h, w = t1.size()

        # print(t1.size())
        t1_feature = self.t1_net(t1, prob)
        t2f_feature = self.t2f_net(t2f, prob)
        
        feature = torch.cat([t1_feature, t2f_feature], dim=1) # b,256,8,8,8    
        # print(feature.size())
        feature = self.ca(feature) + self.sa(feature)
        
        x = torch.flatten(feature, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)        

        return out