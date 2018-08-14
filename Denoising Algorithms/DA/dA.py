import torch
import torch.nn as nn
from torch.autograd import Variable
from time import time
from loadData import loadData
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import numpy as np
from cmath import sqrt


class AutoEncoder(nn.Module):
    def __init__(self,inputsize,outputsize):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inputsize, outputsize),
            nn.ReLU(),

        )
        self.decoder = nn.Sequential(
            nn.Linear(outputsize, inputsize),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def dA(data,inputsize,outputsize,height,width):
    
    
        t0                      = time()
    
        autoencoder             = AutoEncoder(inputsize,outputsize)
        data                    = torch.from_numpy(data).float()
        data                    = Variable(data.view(-1, inputsize))
        
        autoencoder.load_state_dict(torch.load('dA.pt'))
        
        encoded,decoded         = autoencoder(data)
        
        
        dt                      = time() - t0
        
        print('Denoising Autoencoder done in %.2fs.' % dt)
        
        return encoded,decoded
        