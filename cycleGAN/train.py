import numpy as np
import argparse
import os
# import random
import torch
# import torch.backends.cudnn as cudnn
import scipy.io as sio
from torch import autograd
from torch.utils.data import DataLoader
from network.model import Generator, Discriminator, Vgg16
from dataset.dataset import Train_Data, Valid_Data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser()
parser.add_argument('--DIREC', type=str,
                    default='modality_transfer_lsgan', help='project name')
parser.add_argument('--continue_train', type=bool,
                    default=False, help='if load previous model')
parser.add_argument('--max_epoch', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--D_iter', type=int,
                    default=1, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=999, help='random seed')
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

args = parser.parse_args()

# def calc_gradient_penalty(netD, real_data, fake_data, gpu):
#     # print "real_data: ", real_data.size(), fake_data.size()
#     c,s,h,w = real_data.shape
#     alpha = torch.rand(c, s, h, w)
#     alpha = alpha.expand(real_data.size())
#     alpha = alpha.cuda(gpu) if torch.cuda.is_available() else alpha

#     interpolates = alpha * real_data + ((1 - alpha) * fake_data)

#     interpolates = interpolates.cuda(gpu)
#     interpolates = autograd.Variable(interpolates, requires_grad=True)

#     disc_interpolates = netD(interpolates)

#     gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
#                               grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if torch.cuda.is_available() else torch.ones(
#                                   disc_interpolates.size()),
#                               create_graph=True, retain_graph=True, only_inputs=True)[0]
#     gradients = gradients.view(gradients.size(0), -1)
#     gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
#     gradient_penalty = ((gradients_norm - 1) ** 2).mean()
#     # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#     return gradient_penalty

def concatanate_normalize(x):
    return torch.cat(((x-0.485)/0.229, (x-0.456)/0.224, (x-0.406)/0.225), 1)

def trainer(gpu, ngpus_per_node, args): 
    args.gpu = gpu
    
    netG_A2B = Generator()
    netG_B2A = Generator()
    netD_A = Discriminator()
    netD_B = Discriminator()
    vgg = Vgg16()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            netG_A2B.cuda(args.gpu)
            netG_B2A.cuda(args.gpu)
            netD_A.cuda(args.gpu)
            netD_B.cuda(args.gpu)
            vgg.cuda(args.gpu)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            netG_A2B = torch.nn.parallel.DistributedDataParallel(netG_A2B, device_ids=[args.gpu])
            netG_B2A = torch.nn.parallel.DistributedDataParallel(netG_B2A, device_ids=[args.gpu])
            netD_A = torch.nn.parallel.DistributedDataParallel(netD_A, device_ids=[args.gpu])
            netD_B = torch.nn.parallel.DistributedDataParallel(netD_B, device_ids=[args.gpu])
            # vgg = torch.nn.parallel.DistributedDataParallel(vgg, device_ids=[args.gpu])
        else:
            netG_A2B.cuda()
            netG_B2A.cuda()
            netD_A.cuda()
            netD_B.cuda()
            vgg.cuda()
            netG_A2B = torch.nn.parallel.DistributedDataParallel(netG_A2B)
            netG_B2A = torch.nn.parallel.DistributedDataParallel(netG_B2A)
            netD_A = torch.nn.parallel.DistributedDataParallel(netD_A)
            netD_B = torch.nn.parallel.DistributedDataParallel(netD_B)
            vgg = torch.nn.parallel.DistributedDataParallel(vgg)

    
    mse = torch.nn.MSELoss()
    
    params = list(netG_A2B.parameters()) + list(netG_B2A.parameters())
    optimizer_G = torch.optim.Adam(params, lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    g_ls = []
    d_ls = []
    
    adv_ls = []
    cycle_ls = []
    vcycle_ls = []
        
    if args.continue_train:
        checkpoint = torch.load('./save_model/' + args.DIREC + '/model_latest.pkl')
        netG_A2B.load_state_dict(checkpoint['G1'])
        netG_B2A.load_state_dict(checkpoint['G2'])
        netD_A.load_state_dict(checkpoint['D1'])
        netD_B.load_state_dict(checkpoint['D2'])
        restore_epoch = checkpoint['epoch']
        optimizer_G.load_state_dict(checkpoint['opG'])
        optimizer_D_A.load_state_dict(checkpoint['opD1'])
        optimizer_D_B.load_state_dict(checkpoint['opD2'])
        
        readmat = sio.loadmat('./save_loss/' + args.DIREC)
        load_g_loss = readmat['g']
        load_d_loss = readmat['d']
        load_adv_loss = readmat['adv']
        load_cycle_loss = readmat['cycle']
        load_vcycle_loss = readmat['vcycle']
        for i in range(restore_epoch):
            g_ls.append(load_g_loss[0][i])
            d_ls.append(load_d_loss[0][i])
            adv_ls.append(load_adv_loss[0][i])
            cycle_ls.append(load_cycle_loss[0][i])
            vcycle_ls.append(load_vcycle_loss[0][i])
        print('Finish loading loss!')
    else:
        restore_epoch = 0
            
    for epoch_num in range(restore_epoch, args.max_epoch):
        tr_train = Train_Data()
        tr_sampler = DistributedSampler(tr_train)
        trainloader = DataLoader(tr_train, batch_size=args.batch_size, num_workers=args.workers, 
                                 pin_memory=True, sampler=tr_sampler)
        va_train = Valid_Data()
        va_sampler = DistributedSampler(va_train, shuffle=False)   
        validloader = DataLoader(va_train, batch_size=2, num_workers=args.workers, 
                                  pin_memory=True, sampler=va_sampler)

        tr_sampler.set_epoch(epoch_num)
        
        tmp_g_loss = 0.
        tmp_d_loss = 0.
        tmp_adv_loss = 0.
        tmp_cycle_loss = 0.
        tmp_vcycle_loss = 0.          
        
        for i_batch, (x, y) in enumerate(trainloader):
            per_A_loss = 0.
            per_B_loss = 0.
                        
            ##############################################
            # (1) Update G network: Generators A2B and B2A
            ##############################################       
            for p in netD_A.parameters():
                p.requires_grad = False    
            for p in netD_B.parameters():
                p.requires_grad = False    
                
            netG_A2B.train(mode=True)
            netG_B2A.train(mode=True)
                        
            real_image_A = autograd.Variable(x.cuda(args.gpu))
            real_image_B = autograd.Variable(y.cuda(args.gpu))
            
            fake_image_A = netG_B2A(real_image_B)
            fake_output_A = netD_A(fake_image_A)
            loss_GAN_B2A = torch.mean((fake_output_A-1)**2)

            fake_image_B = netG_A2B(real_image_A)
            fake_output_B = netD_B(fake_image_B)
            loss_GAN_A2B = torch.mean((fake_output_B-1)**2)
            
            recovered_image_A = netG_B2A(fake_image_B)
            real_A_feature = vgg(concatanate_normalize(real_image_A))
            fake_A_feature = vgg(concatanate_normalize(recovered_image_A))
            for per_num in range(4):
                per_A_loss += 0.25 * mse(fake_A_feature[per_num], real_A_feature[per_num])
            loss_cycle_ABA = per_A_loss
    
            recovered_image_B = netG_A2B(fake_image_A)
            real_B_feature = vgg(concatanate_normalize(real_image_B))
            fake_B_feature = vgg(concatanate_normalize(recovered_image_B))
            for per_num in range(4):
                per_B_loss += 0.25 * mse(fake_B_feature[per_num], real_B_feature[per_num])
            loss_cycle_BAB = per_B_loss
    
            errG = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
    
            optimizer_G.zero_grad()
            errG.backward()
            optimizer_G.step()

            ##############################################
            # (2) Update D network: Discriminator A
            ##############################################
            for p in netD_A.parameters():
                p.requires_grad = True
            
            for iter_d in range(args.D_iter):
                                
                real_image_A = autograd.Variable(x.cuda(args.gpu))                
                real_output_A = netD_A(real_image_A)
                errD_real_A = 0.5 * torch.mean((real_output_A-1)**2)
                
                with torch.no_grad():
                    real_image_B = autograd.Variable(y.cuda(args.gpu))  
                fake_image_A = netG_B2A(real_image_B)                    
                fake_output_A = netD_A(fake_image_A.data)
                errD_fake_A = 0.5 * torch.mean((fake_output_A-0)**2)
                
                errD_A = errD_real_A + errD_fake_A
        
                optimizer_D_A.zero_grad()
                errD_A.backward()
                optimizer_D_A.step()
    
            ##############################################
            # (3) Update D network: Discriminator B
            ##############################################
            for p in netD_B.parameters():
                p.requires_grad = True
            
            for iter_d in range(args.D_iter):
        
                real_image_B = autograd.Variable(y.cuda(args.gpu))                
                real_output_B = netD_B(real_image_B)
                errD_real_B = 0.5 * torch.mean((real_output_B-1)**2)
                      
                with torch.no_grad():
                    real_image_A = autograd.Variable(x.cuda(args.gpu))  
                fake_image_B = netG_A2B(real_image_A)                    
                fake_output_B = netD_B(fake_image_B.data)
                errD_fake_B = 0.5 * torch.mean((fake_output_B-0)**2)
                
                errD_B = errD_real_B + errD_fake_B
        
                optimizer_D_B.zero_grad()
                errD_B.backward()
                optimizer_D_B.step()
                
            if (i_batch+1) % 10 == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                    print('[Epoch: %d/%d, Batch: %d/%d] G: %.4f, D: %.4f, adv: %.4f, cyc: %.4f,' % 
                          (epoch_num+1, args.max_epoch, i_batch+1, len(trainloader), 
                           errG.item(), (errD_A+errD_B).item(), (loss_GAN_A2B + loss_GAN_B2A).item(), (loss_cycle_ABA + loss_cycle_BAB).item()))
            
            tmp_g_loss += (loss_cycle_ABA + loss_cycle_BAB).item()
            tmp_d_loss += (errD_A+errD_B).item()
            tmp_adv_loss += (loss_GAN_A2B + loss_GAN_B2A).item()
            tmp_cycle_loss += (loss_cycle_ABA + loss_cycle_BAB).item()
        
        
        netG_A2B.eval()
        netG_A2B.train(mode=False)
        netG_B2A.eval()
        netG_B2A.train(mode=False)
        with torch.no_grad():               
            for i_batch, (x, y) in enumerate(validloader):
                real_image_A = autograd.Variable(x.cuda(args.gpu))
                real_image_B = autograd.Variable(y.cuda(args.gpu))
                
                fake_image_A = netG_B2A(real_image_B)   
                fake_image_B = netG_A2B(real_image_A)
        
                recovered_image_A = netG_B2A(fake_image_B)
                real_A_feature = vgg(concatanate_normalize(real_image_A))
                fake_A_feature = vgg(concatanate_normalize(recovered_image_A))
                for per_num in range(4):
                    per_A_loss += 0.25 * mse(fake_A_feature[per_num], real_A_feature[per_num])
                loss_cycle_ABA = per_A_loss
        
                recovered_image_B = netG_A2B(fake_image_A)
                real_B_feature = vgg(concatanate_normalize(real_image_B))
                fake_B_feature = vgg(concatanate_normalize(recovered_image_B))
                for per_num in range(4):
                    per_B_loss += 0.25 * mse(fake_B_feature[per_num], real_B_feature[per_num])
                loss_cycle_BAB = per_B_loss
                
            tmp_vcycle_loss += (loss_cycle_ABA + loss_cycle_BAB).item()

        g_ls.append(tmp_g_loss/len(trainloader))   
        d_ls.append(tmp_d_loss/len(trainloader))   
        adv_ls.append(tmp_adv_loss/len(trainloader))
        cycle_ls.append(tmp_cycle_loss/len(trainloader))   
        vcycle_ls.append(tmp_vcycle_loss/len(validloader))   
        
        sio.savemat('./save_loss/' + args.DIREC +'.mat', {'g': g_ls,'d': d_ls,'adv': adv_ls,
                                                     'cycle': cycle_ls,'vcycle': vcycle_ls})
        if not os.path.exists('./save_model/' + args.DIREC):
                os.makedirs('./save_model/' + args.DIREC) 
                
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
            torch.save({'epoch': epoch_num+1, 'G1': netG_A2B.state_dict(), 'G2': netG_B2A.state_dict(),
                    'D1': netD_A.state_dict(), 'D2': netD_B.state_dict(),
                    'opG': optimizer_G.state_dict(), 'opD1': optimizer_D_A.state_dict(), 'opD2': optimizer_D_B.state_dict()}, 
                    './save_model/' + args.DIREC + '/model_latest.pkl')
            
        if (epoch_num+1) % 10 == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):            
                torch.save({'epoch': epoch_num+1, 'G1': netG_A2B.state_dict(), 'G2': netG_B2A.state_dict(),
                    'D1': netD_A.state_dict(), 'D2': netD_B.state_dict(),
                    'opG': optimizer_G.state_dict(), 'opD1': optimizer_D_A.state_dict(), 'opD2': optimizer_D_B.state_dict()}, 
                    './save_model/' + args.DIREC + '/model_epoch_' + str(epoch_num+1) + '.pkl')
                
    print("Training Finished!")


def main():
    
    args = parser.parse_args()    
 
    if not os.path.exists('./save_model/' + args.DIREC):
        os.makedirs('./save_model/' + args.DIREC)     
 
    # seed = args.seed
    # cudnn.deterministic = True
    # random.seed(seed)
    # # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(trainer, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        trainer(args.gpu, ngpus_per_node, args)

if __name__ == "__main__":
    main()