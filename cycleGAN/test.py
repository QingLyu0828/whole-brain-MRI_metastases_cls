import argparse
import os
import numpy as np
import h5py
import torch
import xlrd
import nibabel as nib
from torch.utils.data import DataLoader
from dataset.dataset import Test_T1
from network.model import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--DIREC', type=str,
                    default='modality_transfer_lsgan', help='project name')
parser.add_argument('--restore_epoch', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
args = parser.parse_args()


device = torch.device('cuda:0')

def norm(img):
    return (img-np.amin(img))/(np.amax(img)-np.amin(img))

def inference(args, netG_A2B, restore_epoch):    
	# read the excel to catch data info
    xlspath = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/labeled_nii_info.xls'
    rb = xlrd.open_workbook(xlspath)
    sheet = rb.sheet_by_index(0)    
    
    netG_A2B = netG_A2B.to(device)    

    batch_size = args.batch_size
    db_test = Test_T1()
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False)
	
	# for inference, no training
    netG_A2B.eval()  
    netG_A2B.train(mode=False)
        
    count = 0
    next1 = 0
    count_idx = 0
    this_idx = 0
    with torch.no_grad():               
        for i_batch, (x, name) in enumerate(testloader):
            c,s,h,w = x.size()
            x = x.to(device)
                    
            pred = netG_A2B(x)
            img = np.array(pred.data.squeeze().cpu())
            img = norm(img)
            
            if count >= sheet.cell_value(count_idx,4): # move to a new case, create save data
                prob = np.zeros((int(sheet.cell_value(count_idx,1)), h, w), dtype=np.float32)
                if count_idx > 0: # save the previous batch of data belongs to the new case
                    if next1 > 0:
                        prob[0:next1,:,:] = prev_tmp2
                count_idx += 1
                
            if count+c >= sheet.cell_value(count_idx,4): # move to the next case, save this case
                filename = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/T1_nii/Image/' + str(name[0])
                hdr = nib.load(filename) 
                
                next1 = int(count+c-sheet.cell_value(count_idx,4))
                remaining = int(c - next1)
                this_idx = next1
                
                tmp2 = np.single(img)
                prob[int(sheet.cell_value(count_idx-1,1))-remaining:int(sheet.cell_value(count_idx-1,1)),:,:] = tmp2[0:remaining,:,:]
                nout2 = np.transpose(prob,(1,2,0))
                out2 = nib.Nifti1Image(nout2, affine=hdr.affine)
                out2.header.get_xyzt_units()
                name2 = '/gpfs/u/scratch/DTIR/DTIRqngl/Data/WFU/T1_nii/cycleGAN/' + str(name[0])
                out2.to_filename(name2) # save data in nii.gz file
                print(name2)
                
                if next1 > 0: # save data belongs to the next case 
                    prev_tmp2 = tmp2[remaining:c,:,:]
            else:
                prob[this_idx:this_idx+c,:,:] = np.single(img)
                this_idx += c 
                
            count += c
    
    print("Testing Finished!")


if __name__ == "__main__":
    continue_train = False
    restore_epoch = args.restore_epoch
    
    netG_A2B = Generator() # T1CE to FSPGR
    netG_B2A = Generator() # FSPGR to T1CE
    
    netG_A2B = torch.nn.DataParallel(netG_A2B)
    netG_B2A = torch.nn.DataParallel(netG_B2A)
    
    checkpoint = torch.load('./save_model/' + args.DIREC + '/model_epoch_' + str(restore_epoch) + '.pkl', map_location='cpu')
    netG_A2B.load_state_dict(checkpoint['G1'])

    inference(args, netG_A2B, restore_epoch)