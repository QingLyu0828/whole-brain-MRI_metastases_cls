import argparse
import os
import numpy as np
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import random
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from utils import DiceLoss
import scipy.io as sio
from dataset.dataset import Train_Data, Valid_Data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from network.transformer_encoder import VisionTransformer as ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_epoch', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--restore_epoch', type=int,
                    default=0, help='epoch number to start the training')
parser.add_argument('--continue_train', type=bool,
                    default=False, help='if load previous model')
parser.add_argument('--DIREC', type=str,
                    default='segmentation_CE_Dice_t2f', help='project name')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=4, help='using number of skip-connect, default is num')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--nodes", type=int, default=1, help='number of nodes')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                          'N processes per node, which has N GPUs. This is the '
                          'fastest way to use PyTorch for either single node or '
                          'multi node data parallel training')

def trainer(gpu, ngpus_per_node, args):
    
    args.gpu = gpu

    model = ViT_seg(img_size=args.img_size)
    # print(model)
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            print('rank =', args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.continue_train:
        checkpoint = torch.load('./save_model/' + args.DIREC + '/model_epoch_' + str(args.restore_epoch) + '.pkl')
        model.load_state_dict(checkpoint['net'])
        model.eval()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)   
    
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    
    tr_train = Train_Data()
    tr_sampler = DistributedSampler(tr_train)
    trainloader = DataLoader(tr_train, batch_size=args.batch_size, num_workers=args.workers, 
                             pin_memory=True, sampler=tr_sampler)
    va_train = Valid_Data()
    va_sampler = DistributedSampler(va_train, shuffle=False)   
    validloader = DataLoader(va_train, batch_size=2, num_workers=args.workers, 
                             pin_memory=True, sampler=va_sampler)   

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999))
    if args.continue_train:
        optimizer.load_state_dict(checkpoint['op'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
	# save values to monitor the training process
    ce_ls = [] # training cross-entropy in each epoch
    dice_ls = [] # training dice score in each epoch
    ls = [] # training loss in each epoch
    vce_ls = [] # validation cross-entropy in each epoch
    vdice_ls = [] # validation dice score in each epoch
    vls = [] # validation loss in each epoch
    if args.continue_train: # continue training and load previously record losses
        readmat = sio.loadmat('./save_loss/' + args.DIREC)
        load_ce_loss = readmat['ce']
        load_dice_loss = readmat['dice']
        load_loss = readmat['loss']
        load_vce_loss = readmat['vce']
        load_vdice_loss = readmat['vdice']
        load_vloss = readmat['vloss']
        for i in range(args.restore_epoch):
            ce_ls.append(load_ce_loss[0][i])
            dice_ls.append(load_dice_loss[0][i])
            ls.append(load_loss[0][i])
            vce_ls.append(load_vce_loss[0][i])
            vdice_ls.append(load_vdice_loss[0][i])
            vls.append(load_vloss[0][i])
        print('Finish loading loss!')
                        
    for epoch_num in range(args.restore_epoch, args.max_epoch):
        tr_sampler.set_epoch(epoch_num)
        
        if epoch_num >= 50:
            scheduler.step()
            
        count = 0
        countv = 0
        tmp_ce_loss = 0.
        tmp_dice_loss = 0.
        tmp_loss = 0.
        tmp_vce_loss = 0.
        tmp_vdice_loss = 0.
        tmp_vloss = 0.
        
        for i_batch, (x, y) in enumerate(trainloader):
            # print(x.size(), y.size())
            x, y = x.cuda(args.gpu, non_blocking=True), y.cuda(args.gpu, non_blocking=True)
            outputs = model(x)
            loss_ce = ce_loss(outputs, y[:].long())
            loss_dice = dice_loss(outputs, y, softmax=True)
            loss = 1. * loss_ce + 1. * loss_dice # training loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1

            if (i_batch+1) % 10 == 0:
                print('[Epoch: %d/%d, Batch: %d/%d] loss_ce: %.4f, loss_dice: %.4f' % 
                      (epoch_num+1, args.max_epoch, i_batch+1, len(trainloader), loss_ce.item(), loss_dice.item()))
            
            tmp_ce_loss += loss_ce.item()
            tmp_dice_loss += loss_dice.item()
            tmp_loss += loss.item()
		
		# conduct validation each epoch
        for i_batch, (x, y) in enumerate(validloader):
            x, y = x.cuda(args.gpu, non_blocking=True), y.cuda(args.gpu, non_blocking=True)
            outputs = model(x)
            loss_ce = ce_loss(outputs, y[:].long())
            loss_dice = dice_loss(outputs, y, softmax=True)
            loss = 1. * loss_ce + 1. * loss_dice

            countv += 1
            tmp_vce_loss += loss_ce.item()
            tmp_vdice_loss += loss_dice.item()
            tmp_vloss += loss.item()
            
            if (i_batch+1) % 10 == 0:
                print('[Epoch: %d/%d, Batch: %d/%d Test]' % 
                      (epoch_num+1, args.max_epoch, i_batch+1, len(validloader)))

        ce_ls.append(tmp_ce_loss/count)   
        dice_ls.append(tmp_dice_loss/count)   
        ls.append(tmp_loss/count)
        vce_ls.append(tmp_vce_loss/countv)   
        vdice_ls.append(tmp_vdice_loss/countv)   
        vls.append(tmp_vloss/countv)
        
		# record losses every epoch in .mat file
        sio.savemat('./save_loss/' + args.DIREC +'.mat', {'ce': ce_ls,'dice': dice_ls,'loss': ls,
                                                     'vce': vce_ls,'vdice': vdice_ls,'vloss': vls})
        
		# save model       
        if (epoch_num+1) % 20 == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
                torch.save({'net': model.state_dict(), 'op': optimizer.state_dict()}, 
                                './save_model/' + args.DIREC + '/model_epoch_'+str(epoch_num+1)+'.pkl')

    print("Training Finished!")


def main():
    
    args = parser.parse_args()    
 
    if not os.path.exists('./save_model/' + args.DIREC):
        os.makedirs('./save_model/' + args.DIREC)     
 
    seed = 999
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(trainer, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        trainer(args.gpu, ngpus_per_node, args)

if __name__ == "__main__":
    main()