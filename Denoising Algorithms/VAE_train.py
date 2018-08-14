from time import time

import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import argparse
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim as ssim


from skimage import io,filters,feature
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

from PIL import Image

from os import listdir


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCHS = 10
BATCH_SIZE = 128
LR = 0.0001         # learning rate
N_TRAIN_IMG = 2
N_VIEW_IMG = 5
MEAN = 0
STDDEV = 0.25
HUNITS = 100
HUNITS_mu = 30
HUNITS_var = 30
height = 510
width = 510
patch_size = 12
log_interval = 1000


path_train = "/users/gillesrobijns/Documents/Thesis/images/Train/"
path_validation = "/users/gillesrobijns/Documents/Thesis/images/Test/"
path_save= "/users/gillesrobijns/Documents/Thesis/dA.pt"

def loadData(path,patchSize,nImages):
    # return array of images
    print('Loading images from' + path_train +'...')
    t0 = time()

    imagesList = listdir(path)
    loadedImages  = np.array([])
    
    for i in range(0,nImages):
        
        if not imagesList[i] == ".DS_Store":
            img             = io.imread(path + imagesList[i],as_grey=True)
            pat             = extract_patches_2d(img[0:height,0:width], (patchSize,patchSize))
            loadedImages    = np.append(loadedImages, pat)
     
    npatches            = pat.shape[0]                 
    loadedImages        = np.reshape(loadedImages,(-1,patchSize,patchSize))       
    loadedPatches       = torch.from_numpy(loadedImages).float()
    
    print('Images loaded in %.2fs.' % (time() - t0))
    
    return loadedPatches, npatches

train_data,npatch  = loadData(path_train,patch_size,N_TRAIN_IMG)

def noise(inp, mean=0, stddev=0.1):
    input_array = inp.numpy()
    
    noisy = input_array + stddev * np.random.standard_normal(inp.shape)
    noisy = np.clip(noisy, 0, 1)

    output_tensor = torch.from_numpy(noisy)
    out = output_tensor.float()
    
    return out

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

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



VAE         = VarAutoEncoder(patch_size*patch_size,HUNITS,HUNITS_mu,HUNITS_var)

optimizer   = torch.optim.Adam(VAE.parameters(), lr=LR)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    
    BCE         = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD         = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    
    VAE.train()
    train_loss  = 0
    
    for batch_idx, data in enumerate(train_loader):
        
        data    = Variable(data.view(-1,patch_size*patch_size))
        
        optimizer.zero_grad()
        
        recon_batch, mu, logvar     = VAE(data)
        loss                        = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss                  += loss.data[0]
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.5f}'.format(
          epoch, 
          train_loss / len(train_loader.dataset)))

def test(epoch):
    
    VAE.eval()
    test_loss   = 0
    
    for i, data in enumerate(test_loader):
        
        if CUDA:
            data = data.cuda()
            
        data                        = Variable(data.view(-1,patch_size*patch_size), volatile=True)
        recon_batch, mu, logvar     = VAE(data)
        test_loss                   += loss_function(recon_batch, data, mu, logvar,patch_size).data[0]
        
        if i == 0:
            
            n           = min(data.size(0), 8)
            comparison  = torch.cat([data[:n],recon_batch.view(args.BATCH_SIZE, 1, patch_size, patch_size)[:n]])
            
            save_image(comparison.data.cpu(),'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    
    print('====> Test set loss: {:.5f}'.format(test_loss))
    
    
for epoch in range(1, EPOCHS + 1):
    
    train(epoch)   


torch.save(VAE.state_dict(), 'VAE.pt')

# initialize figure
f, a = plt.subplots(3, N_VIEW_IMG-1, figsize=(8, 4))  # nrows,ncols,
plt.ion()   # continuously plot
plt.suptitle('Variational Autoencoder', fontsize=16)

# original data (first row) for viewing

validation_data,npatch      = loadData(path_validation, patch_size,N_VIEW_IMG)
view_data                   = Variable(validation_data.view(-1, patch_size*patch_size))
view_noise_data             = Variable(noise(view_data.data,MEAN,0.25))

decoded_data, mu, logvar = VAE(view_noise_data)

view_data                   = np.reshape(view_data.data.numpy(), (-1,patch_size, patch_size))
view_noise_data             = np.reshape(view_noise_data.data.numpy(), (-1,patch_size, patch_size))
decoded_data                = np.reshape(decoded_data.data.numpy(), (-1,patch_size, patch_size))



for i in range(0,N_VIEW_IMG-1):
    
    reconstruction_view         = reconstruct_from_patches_2d(view_data[npatch*(i):npatch*(i+1)], (height,width))
    reconstruction_view_noise   = reconstruct_from_patches_2d(view_noise_data[npatch*(i):npatch*(i+1)], (height,width))
    reconstruction_decoded      = reconstruct_from_patches_2d(decoded_data[npatch*(i):npatch*(i+1)], (height,width))


    a[0][i].imshow(reconstruction_view, cmap='gray')
    a[0][i].axis('off')
    a[0][i].set_title('Original')
    a[1][i].imshow(reconstruction_view_noise, cmap='gray')
    a[1][i].axis('off')
    a[1][i].set_title('Noisy') 
    a[2][i].imshow(reconstruction_decoded, cmap='gray')
    a[2][i].axis('off')
    a[2][i].set_title('Denoised')    



    psnr    = compare_psnr(view_data[npatch*(i):npatch*(i+1)], decoded_data[npatch*(i):npatch*(i+1)])
    sim     = ssim(view_data[npatch*(i):npatch*(i+1)], decoded_data[npatch*(i):npatch*(i+1)])
    
    print('PSNR image ', i+1 , ': ', psnr)
    print('SSIM image ', i+1 , ': ', sim)                                         
            


                    
plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()