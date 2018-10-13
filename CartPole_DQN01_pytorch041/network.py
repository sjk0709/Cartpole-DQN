#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 16:38:27 2018

@author: song
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 20:49:19 2018

@author: song
"""

# -*- coding: utf-8 -*-
"""
Created on Tue April 3 10:56:53 2018

Convolutional VAriational Autoencode

@author: Jaekyung Song
"""

import sys, os
sys.path.append(os.pardir)  # parent directory
#import numpy as np
#import time

#import torch
import torch.nn as nn
#import torch.optim as optim
#import torch.nn.init as init
#import torch.nn.functional as F
#import torchvision.datasets as dset
#import torchvision.transforms as transforms
#from torch.utils.data import TensorDataset
#from torch.utils.data import DataLoader
#from torch.autograd import Variable



# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)      
# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
    
class Network0(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network0, self).__init__()
        
        self.fc1 = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.BatchNorm1d(128), 
                nn.ReLU(), 
#                nn.BatchNorm1d(10), 
                
                nn.Linear(128, 256),
                nn.BatchNorm1d(256), 
                nn.ReLU(),
#                nn.BatchNorm1d(10), 

                nn.Linear(256, 512),
                nn.BatchNorm1d(512), 
                nn.ReLU(),
#        
#                nn.Linear(16, 16 ),
#                nn.ReLU(), 
#                nn.Tanh(),
#                nn.BatchNorm1d(16),                                                              
#                nn.Dropout(),
                
#                nn.Linear(32, 64 ),
#                nn.BatchNorm1d(64), 
#                nn.ReLU(),     
#                nn.Tanh(),
#                nn.BatchNorm1d(10),                                           
#                nn.Dropout(),
      
                nn.Linear(512, output_size),
#                nn.Softmax(1),
#                nn.Sigmoid()                       
        )
         
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, x):        
#        print(x)
        out = self.fc1(x) 
#        print(out)
#        out = self.sigmoid(out)
        
        return out



