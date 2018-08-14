from torch.autograd import Variable
from time import time
import torch.nn as nn
import torch


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
    

def sdA2(data,inputsize,midsize,outputsize):
    
        t0                      = time()
    
        autoencoder_1           = AutoEncoder(inputsize,midsize)
        autoencoder_2           = AutoEncoder(midsize,outputsize)
        
        autoencoder_1.load_state_dict(torch.load('sdA2_1.pt'))
        autoencoder_2.load_state_dict(torch.load('sdA2_2.pt'))
        
        data                    = torch.from_numpy(data).float()
        data                    = Variable(data.view(-1, inputsize))
        
        
        encoded_1               = autoencoder_1.encoder(data)
        encoded_2               = autoencoder_2.encoder(encoded_1)
        
        decoded_2               = autoencoder_2.decoder(encoded_2)
        decoded_1               = autoencoder_1.decoder(decoded_2)
        
        dt                      = time() - t0
        
        print('Stacked Denoising Autoencoder (2 layers) done in %.2fs.' % dt)
        
        return encoded_2,decoded_1
    
def sdA3(data,inputsize,midsize1,midsize2,outputsize):
    
        t0                      = time()
    
        autoencoder_1           = AutoEncoder(inputsize,midsize1)
        autoencoder_2           = AutoEncoder(midsize1,midsize2)
        autoencoder_3           = AutoEncoder(midsize2,outputsize)
        
        autoencoder_1.load_state_dict(torch.load('sdA3_1.pt'))
        autoencoder_2.load_state_dict(torch.load('sdA3_2.pt'))
        autoencoder_3.load_state_dict(torch.load('sdA3_3.pt'))
        
        data                    = torch.from_numpy(data).float()
        data                    = Variable(data.view(-1, inputsize))
        
        
        encoded_1               = autoencoder_1.encoder(data)
        encoded_2               = autoencoder_2.encoder(encoded_1)
        encoded_3               = autoencoder_3.encoder(encoded_2)
        
        decoded_3               = autoencoder_3.decoder(encoded_3)
        decoded_2               = autoencoder_2.decoder(decoded_3)
        decoded_1               = autoencoder_1.decoder(decoded_2)
        
        dt                      = time() - t0
        
        print('Stacked Denoising Autoencoder (3 layers) done in %.2fs.' % dt)
        
        return encoded_3,decoded_1