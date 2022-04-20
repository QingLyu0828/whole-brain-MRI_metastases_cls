import argparse
import os
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from dataset.dataset import Test_Data
from network.transformer_encoder import VisionTransformer as ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=4, help='using number of skip-connect, default is num')
args = parser.parse_args()


device = torch.device('cuda:0')

def inference(args, model, restore_epoch):
    out = np.zeros((200, 2, 512, 512))
    
    model = model.to(device)    
        
    sm = torch.nn.Softmax()

    batch_size = args.batch_size * args.n_gpu
    db_test = Test_Data()
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False)

    model.eval()  
        
    count = 0
    for i_batch, (x, y) in enumerate(testloader):
        c,s,h,w = x.size()
        x = x.to(device)
                
        outputs = model(x)
        # print(outputs.size())       
        pred = sm(outputs)
        # print(pred.size())
        # _, predicted = torch.max(pred, 1)        
        # print(predicted.size())
        # out[count:count+c,:,:] = predicted.data.squeeze().cpu()
        # print(np.amin(out[count:count+c,:,:]), np.amax(out[count:count+c,:,:]))
        count += c
        print(count)
        
        out[count-c:count,:,:,:] = pred.data.squeeze().cpu()
        
    if not os.path.exists('output/' + DIREC):
        os.makedirs('output/' + DIREC)
        
    path = 'output/' + DIREC + '/' + repr(restore_epoch) + '.hdf5'
    f = h5py.File(path, 'w')
    f.create_dataset('data', data=out)
    f.close()
    
    print("Validation Finished!")


if __name__ == "__main__":
    continue_train = False
    restore_epoch = 100
    DIREC = 'segmentation_CE_Dice_t2f'
    
    net = ViT_seg(img_size=args.img_size)
    
    net = torch.nn.DataParallel(net)
    
    checkpoint = torch.load('./save_model/' + DIREC + '/model_epoch_' + str(restore_epoch) + '.pkl', map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    # optimizer.load_state_dict(checkpoint['op'])

    inference(args, net, restore_epoch)


