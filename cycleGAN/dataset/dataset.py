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

        self.transform=transforms.Compose([RandomGenerator(output_size=[512, 512])])
        
        # seed = random.randint(0,999)
        # np.random.seed(seed)
        self.indexx = np.random.choice(5921, 2000, replace=False)
        
        self.len = 2000        
        # self.len = 5921
        
    def __getitem__(self, idxx):
        idx = self.indexx[idxx]
        
        xlspath = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/paired_t1_t2f_info.xls'
        rb = xlrd.open_workbook(xlspath)
        sheet = rb.sheet_by_index(0)
        for i in range(1,117):
            if idx >= 53:
                if idx >= sheet.cell_value(i-1,2) and idx < sheet.cell_value(i,2):
                    index = i
            else:
                index = 0
        slice_index = int(idx - sheet.cell_value(index,2))
        
        imgfilename = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/T1_nii/Image/' + sheet.cell_value(index,0)        
        tmp1 = nib.load(imgfilename).get_fdata()
        tmp1 = np.array(tmp1)
        h,w = tmp1.shape[0:2]
        
        labelfilename = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/T2F_nii/Image/' + sheet.cell_value(index,1)        
        tmp2 = nib.load(labelfilename).get_fdata()
        tmp2 = np.array(tmp2)
        c = tmp2.shape[2]
        
        img1 = tmp1[:,:,slice_index]
        tmp = 0
        img2_index = 0
        for kk in range(c):
            img2 = tmp2[:,:,kk]
            hist_2d, _, _ = np.histogram2d(img1.ravel(),img2.ravel(),bins=20)
            mi = self.mutual_information(hist_2d)
            if mi > tmp:
                tmp = mi
                img2_index = kk
        img2 = tmp2[:,:,img2_index]
            
        x = np.zeros((1, h, w))
        y = np.zeros((1, h, w))
        
        x = img1
        y = img2
        
        x = self.norm(x)
        y = self.norm(y)
        
        sample = {'img1': x,'img2': y}
        if self.transform:
            sample = self.transform(sample)  
        x, y = sample['img1'], sample['img2']
        
        data = np.zeros((1, h, w))
        data[0,:,:] = x.copy()
        label = np.zeros((1, h, w))
        label[0,:,:] = y.copy()
        
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        
        data = data.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        
        return data, label
    
    def __len__(self):
        return self.len
    
    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x
    
    def mutual_information(self, hgram):
        pxy = hgram / float(np.sum(hgram))
        px = np.sum(pxy, axis=1) # marginal for x over y
        py = np.sum(pxy, axis=0) # marginal for y over x
        px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
        nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))    
        

class Valid_Data(Dataset):
    def __init__(self):       

        # self.transform=transforms.Compose([RandomGenerator(output_size=[512, 512])])
        
        # seed = random.randint(0,999)
        # np.random.seed(seed)
        self.indexx = np.random.choice(530, 200, replace=False)
        
        self.len = 200
        # self.len = 530
        
    def __getitem__(self, idxx):
        idx = self.indexx[idxx]        
        idx = idx + 5921
        
        xlspath = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/paired_t1_t2f_info.xls'
        rb = xlrd.open_workbook(xlspath)
        sheet = rb.sheet_by_index(0)
        
        for i in range(118, sheet.nrows):
            if idx >= 5974:
                if idx >= sheet.cell_value(i-1,2) and idx < sheet.cell_value(i,2):
                    index = i
            else:
                index = 117
        slice_index = int(idx - sheet.cell_value(index,2))
        
        imgfilename = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/T1_nii/Image/' + sheet.cell_value(index,0)        
        tmp1 = nib.load(imgfilename).get_fdata()
        tmp1 = np.array(tmp1)
        h,w = tmp1.shape[0:2]
        
        labelfilename = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/T2F_nii/Image/' + sheet.cell_value(index,1)        
        tmp2 = nib.load(labelfilename).get_fdata()
        tmp2 = np.array(tmp2)
        c = tmp2.shape[2]
        
        img1 = tmp1[:,:,slice_index]
        tmp = 0
        img2_index = 0
        for kk in range(c):
            img2 = tmp2[:,:,kk]
            hist_2d, _, _ = np.histogram2d(img1.ravel(),img2.ravel(),bins=20)
            mi = self.mutual_information(hist_2d)
            if mi > tmp:
                tmp = mi
                img2_index = kk
        img2 = tmp2[:,:,img2_index]
            
        x = np.zeros((1, h, w))
        y = np.zeros((1, h, w))
        
        x = img1
        y = img2
        
        x = self.norm(x)
        y = self.norm(y)
        
        data = np.zeros((1, h, w))
        data[0,:,:] = x.copy()
        label = np.zeros((1, h, w))
        label[0,:,:] = y.copy()
        
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        
        data = data.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        
        return data, label
    
    def __len__(self):
        return self.len
    
    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x
    
    def mutual_information(self, hgram):
        pxy = hgram / float(np.sum(hgram))
        px = np.sum(pxy, axis=1) # marginal for x over y
        py = np.sum(pxy, axis=0) # marginal for y over x
        px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
        nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))    
   

class Test_T1(Dataset):
    def __init__(self):
        
        xlspath = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/T1_nii_info.xls'
        rb = xlrd.open_workbook(xlspath)
        self.sheet = rb.sheet_by_index(0)
        # self.len = 7079
        self.len = 7969
        
    def __getitem__(self, idx):
      
        for i in range(self.sheet.nrows-1):
            if idx >= self.sheet.cell_value(i,5) and idx < self.sheet.cell_value(i+1,5):
                index = i
                slice_index = idx - self.sheet.cell_value(i,5)
                break
        
        if idx >= self.sheet.cell_value(self.sheet.nrows-1,5):
            index = self.sheet.nrows-1
            slice_index = idx - self.sheet.cell_value(self.sheet.nrows-1,5)    
        
        imgfilename = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/T1_nii/Image/' + self.sheet.cell_value(index,0)        
        tmp1 = nib.load(imgfilename).get_fdata()
        tmp1 = np.array(tmp1)
        h,w = tmp1.shape[0:2]
        
        
        img1 = tmp1[:,:,int(slice_index)]
            
        x = np.zeros((1, h, w))
        
        x = img1
        
        x = self.norm(x)
        
        data = np.zeros((1, h, w))
        data[0,:,:] = x.copy()
        
        data = torch.from_numpy(data)
        
        data = data.type(torch.FloatTensor)
        
        return data, self.sheet.cell_value(index,0)
    
    def __len__(self):
        return self.len
    
    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x