from time import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

import cv2
from skimage import io
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim as ssim

from os import listdir



torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH           = 5
BATCH_SIZE      = 128
LR              = 0.0001         # learning rate
N_TRAIN_IMG     = 2
N_VIEW_IMG      = 5
MEAN            = 0
STDDEV          = 0.32
HUNITS          = 100
height          = 510
width           = 510
patch_size      = 12

path_train          = "/users/gillesrobijns/Documents/Thesis/Images/Train/"
path_validation     = "/users/gillesrobijns/Documents/Thesis/Images/Test/"
path_save           = "/users/gillesrobijns/Documents/Thesis/Denoising Algorithms/dA.pt"

def loadData(path,patchSize,nImages):
    # return array of images
    print('Loading images from' + path_train +'...')
    t0 = time()

    imagesList      = listdir(path)
    loadedImages    = np.array([])
    
    for i in range(0,nImages):
        
        if not imagesList[i] == ".DS_Store":
            img             = io.imread(path + imagesList[i],as_grey=True)
            pat             = extract_patches_2d(img[0:height,0:width], (patchSize,patchSize))
            loadedImages    = np.append(loadedImages, pat)
    
    
    npatches        = pat.shape[0]         
    loadedImages    = np.reshape(loadedImages,(-1,patchSize,patchSize))       
    loadedPatches   = torch.from_numpy(loadedImages).float()
    
    print('Images loaded in %.2fs.' % (time() - t0))
    
    return loadedPatches, npatches

def noise(inp, mean=0, stddev=0.1):
    input_array     = inp.numpy()
    
    noisy           = input_array + stddev * np.random.standard_normal(inp.shape)
    noisy           = np.clip(noisy, 0, 1)

    output_tensor   = torch.from_numpy(noisy)
    out             = output_tensor.float()
    
    return out

train_data, npatch  = loadData(path_train,patch_size,N_TRAIN_IMG)



# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

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


autoencoder = AutoEncoder(patch_size*patch_size,HUNITS)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()


for epoch in range(EPOCH):
    for step, x in enumerate(train_loader):
        b_x = Variable(x.view(-1, patch_size*patch_size))   # batch x, shape (batch, 28*28)
        b_y = Variable(x.view(-1, patch_size*patch_size))   # batch y, shape (batch, 28*28)
        b_n = noise(b_x.data,MEAN,STDDEV)   # apply noise to data
        b_n = Variable(b_n)
        

        encoded, decoded    = autoencoder(b_n)

        loss                = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(b_x), len(train_loader.dataset),
                100. * step / len(train_loader),
                loss.data[0] ))



torch.save(autoencoder.state_dict(), 'dA.pt')

## Testing network

# initialize figure
f, a = plt.subplots(3, N_VIEW_IMG-1, figsize=(8, 4))  # nrows,ncols,
plt.style.use("ggplot")
plt.ion()   # continuously plot
plt.suptitle('Denoising Autoencoder', fontsize=16)


# original data (first row) for viewing

validation_data, npatch     = loadData(path_validation, patch_size,N_VIEW_IMG)
view_data                   = Variable(validation_data.view(-1, patch_size*patch_size))
view_noise_data             = Variable(noise(view_data.data,MEAN,0.25))

_, decoded_data = autoencoder(view_noise_data)

view_data                   = np.reshape(view_data.data.numpy(), (-1,patch_size, patch_size))
view_noise_data             = np.reshape(view_noise_data.data.numpy(), (-1,patch_size, patch_size))
decoded_data                = np.reshape(decoded_data.data.numpy(), (-1,patch_size, patch_size))

for i in range(0,N_VIEW_IMG-1):
    reconstruction_view         = reconstruct_from_patches_2d(view_data[npatch*(i):npatch*(i+1)], (height,width))
    reconstruction_view_noise   = reconstruct_from_patches_2d(view_noise_data[npatch*(i):npatch*(i+1)], (height,width))
    
    
    a[0][i].imshow(reconstruction_view, cmap='gray')
    a[0][i].axis('off')
    a[1][i].imshow(reconstruction_view_noise, cmap='gray')
    a[1][i].axis('off')
    
    reconstruction_decoded = reconstruct_from_patches_2d(decoded_data[npatch*(i):npatch*(i+1)], (height,width))

    a[2][i].imshow(reconstruction_decoded, cmap='gray')
    a[2][i].axis('off')
    
    psnr    = compare_psnr(view_data[npatch*(i):npatch*(i+1)], decoded_data[npatch*(i):npatch*(i+1)])
    sim     = ssim(view_data[npatch*(i):npatch*(i+1)], decoded_data[npatch*(i):npatch*(i+1)])
    
    print('PSNR image ', i+1 , ': ', psnr)
    print('SSIM image ', i+1 , ': ', sim)
      
plt.draw(); plt.pause(0.05)



plt.ioff()
plt.show()
