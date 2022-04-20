import argparse
import os
import numpy as np
import torch
import h5py
from torch import autograd
from torch.nn.modules.loss import CrossEntropyLoss
from dataset.dataset_fold1 import Valid_Data
from torch.utils.data import DataLoader
from network.model_3d import ClassificationNetwork
import xlrd

parser = argparse.ArgumentParser()
parser.add_argument('--DIREC', type=str,
                    default='classify_fold1', help='project name')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
args = parser.parse_args()


device = torch.device('cuda:0')

def inference(args, model, restore_epoch):
    out_pred = np.zeros((158,5))
    out_index = np.zeros((158))
    gt = np.zeros((158))
    
    model = model.to(device)    
        
    sm = torch.nn.Softmax()

    batch_size = args.batch_size * args.n_gpu
    db_test = Valid_Data()
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False)
        
    count = 0
    tmp_loss = 0.

    model.eval()
    model.train(mode=False)
    with torch.no_grad():
        for i_batch, (x1, x2, prob, y) in enumerate(testloader):
            b = x1.size(0)
            x1, x2 = x1.to(device), x2.to(device)
            prob, y = prob.to(device), y.to(device)
            outputs = model(x1,x2,prob).data

            pred = sm(outputs)
            _, predicted = torch.max(pred.data, 1)
            
            out_pred[count:count+b,:] = pred.squeeze().cpu()
            out_index[count:count+b] = predicted.squeeze().cpu()
            gt[count:count+b] = y.data.squeeze().cpu()
            count += b
        
    if not os.path.exists('output/' + args.DIREC):
        os.makedirs('output/' + args.DIREC)
        
    path = 'output/' + args.DIREC + '/' + repr(restore_epoch) + '.hdf5'
    f = h5py.File(path, 'w')
    f.create_dataset('pred', data=out_pred)
    f.create_dataset('index', data=out_index)
    f.create_dataset('gt', data=gt)
    f.close()
    
    print("Test Finished!")
    
    return tmp_loss/count
                        

if __name__ == "__main__":
    
    net = ClassificationNetwork(num_classes=args.num_classes)
    
    net = torch.nn.DataParallel(net)
    
    checkpoint = torch.load('./save_model/' + args.DIREC + '/model_latest.pkl', map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    restore_epoch = checkpoint['epoch']
    
    inference(args, net, restore_epoch)
