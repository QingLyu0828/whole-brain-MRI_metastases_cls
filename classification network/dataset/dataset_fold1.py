import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import xlrd
import nibabel as nib
import random
from skimage import transform
import os

def random_rot(img1,img2,img3):
    
    k = np.random.randint(0, 3)
    img1 = np.rot90(img1, k+1)
    img2 = np.rot90(img2, k+1)
    img3 = np.rot90(img3, k+1)
    
    return img1,img2,img3


def random_flip(img1,img2,img3):
    
    axis = np.random.randint(0, 2)
    img1 = np.flip(img1, axis=axis).copy()
    img2 = np.flip(img2, axis=axis).copy()
    img3 = np.flip(img3, axis=axis).copy()
    
    return img1,img2,img3


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        img1, img2, img3 = sample['img1'], sample['img2'], sample['img3']

        if random.random() > 0.5:
            img1,img2,img3 = random_rot(img1,img2,img3)
        if random.random() > 0.5:
            img1,img2,img3 = random_flip(img1,img2,img3)
        sample = {'img1': img1, 'img2': img2, 'img3': img3}
        
        return sample
    
    
class Train_Data(Dataset):
    def __init__(self):       
        
        self.transform=transforms.Compose([RandomGenerator(output_size=[256, 256])])
        self.len = 600
        # self.len = 7
        # self.files = os.listdir('G:/Data/WFU/T2F_nii/Image')
        
    def __getitem__(self, idx):     
        
        randomidx = random.random()
        
        if randomidx < 0.35:
            index = random.randint(0, 795)
            xlspath = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/New_Ten_Fold/fold1_T1_T2F_nii_info_TR_0.xls'
            rb = xlrd.open_workbook(xlspath)
            sheet = rb.sheet_by_index(0)
        elif randomidx >= 0.35 and randomidx < 0.55:
            index = random.randint(0, 314)
            xlspath = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/New_Ten_Fold/fold1_T1_T2F_nii_info_TR_1.xls'
            rb = xlrd.open_workbook(xlspath)
            sheet = rb.sheet_by_index(0)
        elif randomidx >= 0.55 and randomidx < 0.7:
            index = random.randint(0, 79)
            xlspath = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/New_Ten_Fold/fold1_T1_T2F_nii_info_TR_2.xls'
            rb = xlrd.open_workbook(xlspath)
            sheet = rb.sheet_by_index(0)
        elif randomidx >= 0.7 and randomidx < 0.85:
            index = random.randint(0, 96)
            xlspath = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/New_Ten_Fold/fold1_T1_T2F_nii_info_TR_3.xls'
            rb = xlrd.open_workbook(xlspath)
            sheet = rb.sheet_by_index(0)
        else:
            index = random.randint(0, 135)
            xlspath = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/New_Ten_Fold/fold1_T1_T2F_nii_info_TR_4.xls'
            rb = xlrd.open_workbook(xlspath)
            sheet = rb.sheet_by_index(0)        
            
        if sheet.cell_value(index,4) == 'T1':
            base = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/T1_nii'
            img1filename = base + '/Image/' + sheet.cell_value(index,0)
            probfilename = base + '/Prob/' + sheet.cell_value(index,0)[0:-7] + '_prob.nii.gz'
            img2filename = base + '/cycleGAN/' + sheet.cell_value(index,0)
            
        else:
            base = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/T2F_nii'
            img1filename = base + '/cycleGAN/' + sheet.cell_value(index,0)
            probfilename = base + '/Prob/' + sheet.cell_value(index,0)[0:-7] + '_prob.nii.gz'
            img2filename = base + '/Image/' + sheet.cell_value(index,0)
        
        # img1filename = 'G:/Data/WFU/T2F_nii/cycleGAN/' + self.files[idx]
        # probfilename = 'G:/Data/WFU/T2F_nii/Prob/' + self.files[idx][0:-7] + '_prob.nii.gz'
        # img2filename = 'G:/Data/WFU/T2F_nii/Image/' + self.files[idx]
            
        img1 = nib.load(img1filename).get_fdata()
        img1 = np.array(img1)
        h,w,c = img1.shape
        img1 = transform.resize(img1, (h//2,w//2,c))
        img1 = self.norm(img1)

        img2 = nib.load(img2filename).get_fdata()
        img2 = np.array(img2)
        img2 = transform.resize(img2, (h//2,w//2,c))
        img2 = self.norm(img2)

        
        prob = nib.load(probfilename).get_fdata()
        prob = np.array(prob)
        prob = transform.resize(prob, (h//2,w//2,c))
        h,w,c = prob.shape
        
        sample = {'img1': img1, 'img2': img2, 'img3': prob}
        if self.transform:
            sample = self.transform(sample)  
        img1, img2, prob = sample['img1'], sample['img2'], sample['img3']
        
        if c >= 64:
            thresl = int((c-64)/3)
            thresh = thresl+64
            data1 = np.zeros((1, h, w, 64))
            data1[0,:,:,0:64] = img1[:,:,thresl:thresh]
            data2 = np.zeros((1, h, w, 64))
            data2[0,:,:,0:64] = img2[:,:,thresl:thresh]
            data3 = np.zeros((1, h, w, 64))
            data3[0,:,:,0:64] = prob[:,:,thresl:thresh]           
        else:
            data1 = np.zeros((1, h, w, 64))
            data1[0,:,:,0:c] = img1
            data2 = np.zeros((1, h, w, 64))
            data2[0,:,:,0:c] = img2
            data3 = np.zeros((1, h, w, 64))
            data3[0,:,:,0:c] = prob
        
        data1 = np.transpose(data1, (0,3,1,2))
        data2 = np.transpose(data2, (0,3,1,2))
        data3 = np.transpose(data3, (0,3,1,2))
        
        label = int(sheet.cell_value(index,3))
        # label = 0
        # label = np.zeros((1, 2))
        # if int(sheet.cell_value(idx,3)) == 0:
        #     label[0,0] = 1
        # else:
        #     label[0,1] = 1
        
        data1 = torch.from_numpy(data1)
        data2 = torch.from_numpy(data2)
        data3 = torch.from_numpy(data3)
        label = torch.as_tensor(label)
        # label = torch.from_numpy(label)
        
        data1 = data1.type(torch.FloatTensor)
        data2 = data2.type(torch.FloatTensor)
        data3 = data3.type(torch.FloatTensor)
        label = label.type(torch.LongTensor)
        # label = label.type(torch.FloatTensor)
        
        return data1, data2, data3, label
    
    def __len__(self):
        return self.len
    
    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x

class Valid_Data(Dataset):
    def __init__(self):       

        self.len = 158
        
    def __getitem__(self, index):
    
        xlspath = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/New_Ten_Fold/fold1_T1_T2F_nii_info_TE.xls'
        rb = xlrd.open_workbook(xlspath)
        sheet = rb.sheet_by_index(0)
        
        if sheet.cell_value(index,4) == 'T1':
            base = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/T1_nii'
            img1filename = base + '/Image/' + sheet.cell_value(index,0)
            probfilename = base + '/Prob/' + sheet.cell_value(index,0)[0:-7] + '_prob.nii.gz'
            img2filename = base + '/cycleGAN/' + sheet.cell_value(index,0)
            
        else:
            base = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/T2F_nii'
            img1filename = base + '/cycleGAN/' + sheet.cell_value(index,0)
            probfilename = base + '/Prob/' + sheet.cell_value(index,0)[0:-7] + '_prob.nii.gz'
            img2filename = base + '/Image/' + sheet.cell_value(index,0)
            
        img1 = nib.load(img1filename).get_fdata()
        img1 = np.array(img1)
        h,w,c = img1.shape
        img1 = transform.resize(img1, (h//2,w//2,c))
        img1 = self.norm(img1)

        img2 = nib.load(img2filename).get_fdata()
        img2 = np.array(img2)
        img2 = transform.resize(img2, (h//2,w//2,c))
        img2 = self.norm(img2)

        
        prob = nib.load(probfilename).get_fdata()
        prob = np.array(prob)
        prob = transform.resize(prob, (h//2,w//2,c))
        h,w,c = prob.shape
        
        # sample = {'img1': img1, 'img2': img2, 'img3': prob}
        # if self.transform:
        #     sample = self.transform(sample)  
        # img1, img2, prob = sample['img1'], sample['img2'], sample['img3']
        
        if c >= 64:
            thresl = int((c-64)/3)
            thresh = thresl+64
            data1 = np.zeros((1, h, w, 64))
            data1[0,:,:,0:64] = img1[:,:,thresl:thresh]
            data2 = np.zeros((1, h, w, 64))
            data2[0,:,:,0:64] = img2[:,:,thresl:thresh]
            data3 = np.zeros((1, h, w, 64))
            data3[0,:,:,0:64] = prob[:,:,thresl:thresh]           
        else:
            data1 = np.zeros((1, h, w, 64))
            data1[0,:,:,0:c] = img1
            data2 = np.zeros((1, h, w, 64))
            data2[0,:,:,0:c] = img2
            data3 = np.zeros((1, h, w, 64))
            data3[0,:,:,0:c] = prob
        
        data1 = np.transpose(data1, (0,3,1,2))
        data2 = np.transpose(data2, (0,3,1,2))
        data3 = np.transpose(data3, (0,3,1,2))
        
        label = int(sheet.cell_value(index,3))
        # label = np.zeros((1, 2))
        # if int(sheet.cell_value(idx,3)) == 0:
        #     label[0,0] = 1
        # else:
        #     label[0,1] = 1
        
        data1 = torch.from_numpy(data1)
        data2 = torch.from_numpy(data2)
        data3 = torch.from_numpy(data3)
        label = torch.as_tensor(label)
        # label = torch.from_numpy(label)
        
        data1 = data1.type(torch.FloatTensor)
        data2 = data2.type(torch.FloatTensor)
        data3 = data3.type(torch.FloatTensor)
        label = label.type(torch.LongTensor)
        # label = label.type(torch.FloatTensor)
        
        return data1, data2, data3, label
    
    def __len__(self):
        return self.len
    
    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x
