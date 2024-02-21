"""
    Dataset : https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset#1.jpg
"""

## 라이브러리 추가하기
import argparse

import os, itertools, gc
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.model import *
from dataset.dataset import *
from utils.util import *

import matplotlib.pyplot as plt

from torchvision import transforms


def train(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(float)]

    img_size = args.img_size
    in_channels = args.in_channels

    wgt_cycle = args.wgt_cycle
    wgt_adv = args.wgt_adv
    wgt_gp = args.wgt_gp

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## 디렉토리 생성하기
    result_dir_train = os.path.join(result_dir, 'train')

    if not os.path.exists(result_dir_train):
        os.makedirs(os.path.join(result_dir_train, 'png'))

    ## 네트워크 학습하기
    if mode == 'train':
        # transform
        transform_train = transforms.Compose([
                                            # Resize(shape=(286*2, 286*2, nch)),    # Original Article
                                            # RandomCrop((ny, nx)),
                                              MinMaxScaling(),
                                              Resize((img_size, img_size, in_channels)),    
                                              Normalization(mean=0.5, std=0.5)])

        # Trainset
        dataset_train = Dataset(data_dir=os.path.join(data_dir, "train"), 
                                transform=transform_train, 
                                data_type='both')
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True, persistent_workers=True)


        # etc functions
        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)


    ## Network
    if network == "DualGAN":
        # Generator
        netG_a2b = DualGAN(in_channels=in_channels, out_channels=1).to(device)
        netG_b2a = DualGAN(in_channels=in_channels, out_channels=1).to(device)
        # Discriminator
        netD_a = Discriminator(in_channels=in_channels, out_channels=1).to(device)
        netD_b = Discriminator(in_channels=in_channels, out_channels=1).to(device)

        # Initialize Weights
        init_weights(netG_a2b, init_type='normal', init_gain=0.02)
        init_weights(netG_b2a, init_type='normal', init_gain=0.02)

        init_weights(netD_a, init_type='normal', init_gain=0.02)
        init_weights(netD_b, init_type='normal', init_gain=0.02)

    ## Define Loss function
    fn_cycle = nn.L1Loss().to(device)
    ## Set Optimizer
    optim_G = torch.optim.Adam(itertools.chain(netG_a2b.parameters(), netG_b2a.parameters()), betas=(0.5, 0.999), lr=lr)
    optim_D_A = torch.optim.Adam(netD_a.parameters(), betas=(0.5, 0.999), lr=lr)
    optim_D_B = torch.optim.Adam(netD_b.parameters(), betas=(0.5, 0.999), lr=lr)
    

    ## etc functions
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean

    cmap = "gray"

    ## Tensorboard 를 사용하기 위한 SummaryWriter 설정
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))

    ## 네트워크 학습시키기
    st_epoch = 0

    # TRAIN MODE
    if mode == 'train':
        if train_continue == "on":
            netG_a2b, netG_b2a, \
            netD_a, netD_b, \
            optim_G, optim_D_A, optim_D_B, st_epoch = load(ckpt_dir=ckpt_dir,
                                                        netG_a2b=netG_a2b, netG_b2a=netG_b2a, 
                                                        netD_a=netD_a, netD_b=netD_b,
                                                        optim_G=optim_G, optim_D_A=optim_D_A,optim_D_B=optim_D_B)
        for epoch in range(st_epoch + 1, num_epoch + 1):
            netG_a2b.train()    # Generator
            netG_b2a.train() 
            netD_a.train()      # Discriminator
            netD_b.train()

            loss_G_adv_train, loss_D_train, loss_cycle_a_train, loss_cycle_b_train, loss_G_train  = [], [], [], [], []      # Adverserial Loss / Cycle A Loss / Cycle B Loss / Total Loss

            for batch, data in enumerate(loader_train, 1):

                input_a = data['data_a'].to(device)
                input_b = data['data_b'].to(device)

            # Generator (netG_a2b, netG_b)
                
                # Generate a batch of images
                output_a = netG_b2a(input_b)
                output_b = netG_a2b(input_a)
                # Reconstruction
                recon_a = netG_b2a(output_b)
                recon_b = netG_b2a(output_a)

                # print(f"OUTPUT [A] {output_a.shape}, [B] {output_b.shape}, RECON [A] {recon_a.shape}, [B] {recon_b.shape}")

            # Discriminator (D_A, D_B)
                set_requires_grad([netD_a, netD_b], requires_grad=True)
                optim_D_A.zero_grad()
                optim_D_B.zero_grad()

            # ----------
            # Domain A
            # ----------

                # Discriminator (a batch of images)
                pred_real_a = netD_a(input_a)
                pred_fake_a = netD_a(output_a.detach())
                # print(f"CPU/GPU: [input_a]-{input_a.is_cuda}, [pred_fake_a]-{pred_fake_a.is_cuda} ")
                
                # Loss D_A
                gp_A = compute_gradient_penalty(netD_a, input_a, output_b, device)
                loss_D_A = -torch.mean(pred_real_a) + torch.mean(pred_fake_a) + wgt_gp * gp_A

            # ----------
            # Domain B
            # ----------

                # Discriminate a batch of images
                pred_real_b = netD_b(input_b)
                pred_fake_b = netD_b(output_b.detach())

                # Loss D_B
                gp_B = compute_gradient_penalty(netD_b, input_b, output_a, device)
                loss_D_B = -torch.mean(pred_real_b) + torch.mean(pred_fake_b) + wgt_gp * gp_B

                # Total Loss
                D_loss = loss_D_A + loss_D_B
                D_loss.backward(retain_graph=True)

                optim_D_A.step()
                optim_D_B.step()

                # Generator (netD_a, netD_b)
                set_requires_grad([netD_a, netD_b], requires_grad=False)
                optim_G.zero_grad()

                pred_fake_a = netD_a(output_a)
                pred_fake_b = netD_b(output_b)

                # 1. Adversarial Loss
                G_adv = -torch.mean(pred_fake_a) - torch.mean(pred_fake_b)
                # 2. Cycle Consistency Loss
                loss_cycle_a = fn_cycle(recon_a, input_a)
                loss_cycle_b = fn_cycle(recon_b, input_b)
                cycle_loss = loss_cycle_a + loss_cycle_b

                # 3. Total Loss
                G_loss = wgt_adv * G_adv + wgt_cycle * cycle_loss
                                
                G_loss.backward(retain_graph=True)
                optim_G.step()
                    
                # append Loss value
                loss_G_adv_train += [G_adv.item()]
                loss_cycle_a_train += [loss_cycle_a.item()]
                loss_cycle_b_train += [loss_cycle_b.item()]

                loss_D_train += [D_loss.item()]
                loss_G_train += [G_loss.item()]

 
                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                    "GEN adv : %.4f | " 
                    "DISC : %.4f | "
                    "CYCLE : [a] %.4f [b] %.4f | "
                    "Total : %.4f |"
                    "" %
                    (epoch, num_epoch, batch, num_batch_train,
                    np.mean(loss_G_adv_train),
                    np.mean(loss_D_train),
                    np.mean(loss_cycle_a_train), np.mean(loss_cycle_b_train), 
                    np.mean(loss_G_train)
                    ))

                if batch % 50 == 0:
                    # Tensorboard 저장하기
                    input_a = fn_tonumpy(fn_denorm(input_a, mean=0.5, std=0.5)).squeeze()
                    input_b = fn_tonumpy(fn_denorm(input_b, mean=0.5, std=0.5)).squeeze()                    
                    output_a = fn_tonumpy(fn_denorm(output_a, mean=0.5, std=0.5)).squeeze()
                    output_b = fn_tonumpy(fn_denorm(output_b, mean=0.5, std=0.5)).squeeze()

                    input_a = np.clip(input_a, a_min=0, a_max=1)
                    input_b = np.clip(input_b, a_min=0, a_max=1)
                    output_a = np.clip(output_a, a_min=0, a_max=1)
                    output_b = np.clip(output_b, a_min=0, a_max=1)

                    id = num_batch_train * (epoch - 1) + batch

                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input_a.png' % id), input_a[0], cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input_b.png' % id), input_b[0], cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output_a.png' % id), output_a[0], cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output_b.png' % id), output_b[0], cmap=cmap)

                    writer_train.add_image('input_a', input_a[:,:,:, np.newaxis], id, dataformats='NHWC')
                    writer_train.add_image('input_b', input_b[:,:,:, np.newaxis], id, dataformats='NHWC')
                    writer_train.add_image('output_a', output_a[:,:,:, np.newaxis], id, dataformats='NHWC')
                    writer_train.add_image('output_b', output_b[:,:,:, np.newaxis], id, dataformats='NHWC')


            writer_train.add_scalar('loss_G_adv', np.mean(loss_G_adv_train), epoch)
            writer_train.add_scalar('loss_cycle_a', np.mean(loss_cycle_a_train), epoch)
            writer_train.add_scalar('loss_cycle_b', np.mean(loss_cycle_b_train), epoch)
            writer_train.add_scalar('loss_G_total', np.mean(loss_G_train), epoch)
            writer_train.add_scalar('loss_D_total', np.mean(loss_D_train), epoch)
            
            
            if epoch % 25 == 0 or epoch == num_epoch:
                save(ckpt_dir=ckpt_dir, netG_a2b=netG_a2b, netG_b2a=netG_b2a, netD_a=netD_a, netD_b=netD_b, optim_G=optim_G, optim_D_A=optim_D_A, optim_D_B=optim_D_B, epoch=epoch)
            gc.collect()

        writer_train.close()





    # *---------------------  Test Code 작성 필요
