import os
import numpy as np

import torch
import torch.nn as nn

from models.layers import UNetDown, UNetUp

# dualGAN 
# https://openaccess.thecvf.com/content_ICCV_2017/papers/Yi_DualGAN_Unsupervised_Dual_ICCV_2017_paper.pdf

class DualGAN(nn.Module):       # Genetator
    def __init__(self, in_channels=1, out_channels=1):
        super(DualGAN, self).__init__()
        
        # Encoder > Down Sampling
        self.enc1 = UNetDown(in_channels, 64, normalize=False)
        self.enc2 = UNetDown(64, 128)
        self.enc3 = UNetDown(128, 256)
        self.enc4 = UNetDown(256, 512, dropout=0.5)
        self.enc5 = UNetDown(512, 512, dropout=0.5)
        self.enc6 = UNetDown(512, 512, dropout=0.5)
        self.enc7 = UNetDown(512, 512, dropout=0.5, normalize=False)    
        
        # Decoder > Upsampling
        self.dec1 = UNetUp(512, 512, dropout=0.5)
        self.dec2 = UNetUp(1024, 512, dropout=0.5)
        self.dec3 = UNetUp(1024, 512, dropout=0.5)
        self.dec4 = UNetUp(1024, 256)
        self.dec5 = UNetUp(512, 128)
        self.dec6 = UNetUp(256, 64)

        self.final = nn.Sequential(nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1), nn.Tanh())

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        # Decoder
        d1 = self.dec1(e7, e6)
        d2 = self.dec2(d1, e5)
        d3 = self.dec3(d2, e4)
        d4 = self.dec4(d3, e3)
        d5 = self.dec5(d4, e2)
        d6 = self.dec6(d5, e1)

        output = self.final(d6)
        return output
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_ch, out_ch, normalize=True):
            layers = []
            layers += [ nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1) ]

            if normalize:
                layers += [ nn.BatchNorm2d(out_ch, 0.8) ]
            layers += [ nn.LeakyReLU(0.2, inplace=True) ]
            
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d( (1, 0, 1, 0) ),
            nn.Conv2d(256, out_channels, kernel_size=4)
        )
    
    def forward(self, x):
        
        return self.model(x)
