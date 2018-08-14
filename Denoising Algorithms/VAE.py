import torch
import torch.nn as nn
from torch.autograd import Variable
from time import time

class VarAutoEncoder(nn.Module):
    
    def __init__(self,inputSize,midSize,muSize,varSize):
        super(VarAutoEncoder, self).__init__()
        
        self.fc1        = nn.Linear(inputSize, midSize)
        self.fcmu       = nn.Linear(midSize, muSize)
        self.fcvar      = nn.Linear(midSize, varSize)
        self.fc2        = nn.Linear(muSize, midSize)
        self.fc3        = nn.Linear(midSize, inputSize)

        self.relu       = nn.ReLU()
        self.sigmoid    = nn.Sigmoid()

    def encode(self, x):
        
        h1              = self.relu(self.fc1(x))
        
        return self.fcmu(h1), self.fcvar(h1)

    def reparameterize(self, mu, logvar):
        
        if self.training:
            
            std     = logvar.mul(0.5).exp_()
            eps     = Variable(std.data.new(std.size()).normal_())
            
            return eps.mul(std).add_(mu)
        else:
            
            return mu

    def decode(self, z):
        
        h3      = self.relu(self.fc2(z))
        
        return self.sigmoid(self.fc3(h3))

    def forward(self, x):
        
        mu, logvar  = self.encode(x)
        z           = self.reparameterize(mu, logvar)
        
        return self.decode(z), mu, logvar

def VAE(data,inputSize,midSize,muSize,varSize):
    
        t0                      = time()
    
        VAE                     = VarAutoEncoder(inputSize,midSize,muSize,varSize)
        data                    = torch.from_numpy(data).float()
        data                    = Variable(data.view(-1, inputSize))
        
        VAE.load_state_dict(torch.load('VAE.pt'))
        
        decoded,_,_             = VAE(data)
        
        dt                      = time() - t0
        
        print('Variational Autoencoder done in %.2fs.' % dt)
        
        return decoded