import os
import random
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from torchvision import transforms
import xlrd
import nibabel as nib
from skimage import transform

def random_rot(img1,img2):
    
    k = np.random.randint(0, 3)
    img1 = np.rot90(img1, k+1)
    img2 = np.rot90(img2, k+1)
    
    return img1,img2


def random_flip(img1,img2):
    
    axis = np.random.randint(0, 2)
    img1 = np.flip(img1, axis=axis).copy()
    img2 = np.flip(img2, axis=axis).copy()
    
    return img1,img2


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        img1, img2 = sample['img1'], sample['img2']

        if random.random() > 0.5:
            img1,img2 = random_rot(img1,img2)
        if random.random() > 0.5:
            img1,img2 = random_flip(img1,img2)
        sample = {'img1': img1,'img2': img2}
        
        return sample


class Train_Data(Dataset):
    def __init__(self):       
		# load data stored in hdf5 files
        path = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/Contours/TR_T1_data.hdf5'
        f = h5py.File(path,'r')
        data = f['data']
        self.data = np.array(data)
        self.h, self.w, c = self.data.shape
        
        path = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/Contours/TR_T1_label.hdf5'
        f = h5py.File(path,'r')
        label = f['data']        
        self.label = np.array(label)

        self.transform=transforms.Compose([RandomGenerator(output_size=[self.h, self.w])])
        
        self.index = np.random.choice(c, 2400, replace=False)
        
        self.len = 2400
        
    def __getitem__(self, idx):
        x = np.zeros((1, self.h, self.w))
        y = np.zeros((1, self.h, self.w))
        
        x = self.data[:, :, self.index[idx]]
        y = self.label[:, :, self.index[idx]]
        
        x = self.norm(x)
        
        sample = {'img1': x,'img2': y}
        if self.transform:
            sample = self.transform(sample)  
        x, y = sample['img1'], sample['img2']
        
        data = np.zeros((1, self.h, self.w))
        data[0,:,:] = x.copy()
        label = np.zeros((self.h, self.w))
        label = y.copy()
        
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        
        data = data.type(torch.FloatTensor)
        label = label.type(torch.LongTensor)
        
        return data, label
    
    def __len__(self):
        return self.len
    
    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x
    
    
class Test_Data(Dataset):
    def __init__(self):       
        path = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/Contours/VA_T2F_data.hdf5'
        f = h5py.File(path,'r')
        data = f['data'][:,:,200:250]
        self.data = np.array(data)
        self.h, self.w, c = self.data.shape
        
        path = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/Contours/VA_T2F_label.hdf5'
        f = h5py.File(path,'r')
        label = f['data'][:,:,200:250]
        self.label = np.array(label)

        self.len = c
        
    def __getitem__(self, idx):
        x = np.zeros((1, self.h, self.w))
        y = np.zeros((1, self.h, self.w))
        
        x = self.data[:, :, idx]
        y = self.label[:, :, idx]

        x = self.norm(x)
        
        data = np.zeros((1, self.h, self.w))
        data[0,:,:] = x.copy()
        label = np.zeros((self.h, self.w))
        label = y.copy()
        
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        
        data = data.type(torch.FloatTensor)
        label = label.type(torch.LongTensor)
        
        return data, label
    
    def __len__(self):
        return self.len
    
    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x
        

class Valid_Data(Dataset):
    def __init__(self):       
        path = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/Contours/VA_T1_data.hdf5'
        f = h5py.File(path,'r')
        data = f['data']
        self.data = np.array(data)
        self.h, self.w, c = self.data.shape
        
        path = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/Contours/VA_T1_label.hdf5'
        f = h5py.File(path,'r')
        label = f['data']        
        self.label = np.array(label)

        self.index = np.random.choice(c, 200, replace=False)
        self.len = 200
        
    def __getitem__(self, idx):
        x = np.zeros((1, self.h, self.w))
        y = np.zeros((1, self.h, self.w))
        
        x = self.data[:, :, self.index[idx]]
        y = self.label[:, :, self.index[idx]]

        x = self.norm(x)
        
        data = np.zeros((1, self.h, self.w))
        data[0,:,:] = x.copy()
        label = np.zeros((self.h, self.w))
        label = y.copy()
        
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        
        data = data.type(torch.FloatTensor)
        label = label.type(torch.LongTensor)
        
        return data, label
    
    def __len__(self):
        return self.len
    
    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x