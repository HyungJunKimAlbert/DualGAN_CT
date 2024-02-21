import os
import numpy as np

import torch
import torch.nn as nn
from torchvision.models import vgg19    # pre-trained model

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = []
        layers += [ nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False) ]

        if normalize: # Normalize
            layers += [ nn.InstanceNorm2d(out_channels, affine=True) ]
        layers += [ nn.LeakyReLU(0.2) ]
        if dropout: # Dropout
            layers += [ nn.Dropout2d(dropout) ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = []
        layers +=   [ nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.InstanceNorm2d(out_channels, affine=True),
                      nn.ReLU(inplace=True) ]
        if dropout:   # Dropout
            layers += [ nn.Dropout2d(dropout) ]
        
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat( (x, skip_input), 1 )
        
        return x
        
            

    
