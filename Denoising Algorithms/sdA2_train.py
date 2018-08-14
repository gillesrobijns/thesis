from time import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim as ssim

from os import listdir



torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH                       = 10
BATCH_SIZE                  = 128
LR                          = 0.00001         
N_TRAIN_IMG                 = 2
N_VIEW_IMG                  = 5
MEAN                        = 0
STDDEV                      = 0.25
HUNITS1                     = 200
HUNITS2                     = 150
height                      = 510
width                       = 510
patch_size                  = 12

path_train = "/users/gillesrobijns/Documents/Thesis/images/Train/"
path_validation = "/users/gillesrobijns/Documents/Thesis/images/Test/"
path_save= "/users/gillesrobijns/Documents/Thesis/Denoising Algorithms/sdA2.pt"

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
    input_array = inp.numpy()
    
    noisy = input_array + stddev * np.random.standard_normal(inp.shape)
    noisy = np.clip(noisy, 0, 1)

    output_tensor = torch.from_numpy(noisy)
    out = output_tensor.float()
    
    return out

train_data, npatch  = loadData(path_train,patch_size,N_TRAIN_IMG)



# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader_1 = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

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



autoencoder_1 = AutoEncoder(patch_size*patch_size,HUNITS1)



optimizer_1 = torch.optim.Adam(autoencoder_1.parameters(), lr=LR)
loss_func_1 = nn.MSELoss()


for epoch in range(EPOCH):
    for step, x in enumerate(train_loader_1):
        b_x = Variable(x.view(-1, patch_size*patch_size))   # batch x, shape (batch, 28*28)
        b_y = Variable(x.view(-1, patch_size*patch_size))   # batch y, shape (batch, 28*28)
        b_n = noise(b_x.data,MEAN,STDDEV)   # apply noise to data
        b_n = Variable(b_n)
        

        encoded_1, decoded_1 = autoencoder_1(b_n)

        loss_1 = loss_func_1(decoded_1, b_y)      # mean square error
        optimizer_1.zero_grad()               # clear gradients for this training step
        loss_1.backward()                     # backpropagation, compute gradients
        optimizer_1.step()                    # apply gradients

        if step % 100 == 0:
            print('First layer, Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(b_x), len(train_loader_1.dataset),
                100. * step / len(train_loader_1),
                loss_1.data[0] ))




torch.save(autoencoder_1.state_dict(), 'sdA2_1.pt')

train_data_1 = Variable(train_data.view(-1, patch_size*patch_size))
encoded_1 = autoencoder_1.encoder(train_data_1)


train_loader_2 = Data.DataLoader(dataset=encoded_1.data, batch_size=BATCH_SIZE, shuffle=True)

autoencoder_2 = AutoEncoder(HUNITS1,HUNITS2)
optimizer_2 = torch.optim.Adam(autoencoder_2.parameters(), lr=LR)
loss_func_2 = nn.MSELoss()


for epoch in range(EPOCH):
    for step, x in enumerate(train_loader_2):
        b_x = Variable(x.view(-1, HUNITS1))  
        b_y = Variable(x.view(-1, HUNITS1))  
        b_n = noise(b_x.data,MEAN,STDDEV)   
        b_n = Variable(b_n)
        

        encoded_2, decoded_2 = autoencoder_2(b_n)

        loss_2 = loss_func_2(decoded_2, b_y)      
        optimizer_2.zero_grad()               
        loss_2.backward()                     
        optimizer_2.step()                   

        if step % 100 == 0:
            print('Second layer, Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(b_x), len(train_loader_2.dataset),
                100. * step / len(train_loader_2),
                loss_2.data[0] ))





torch.save(autoencoder_2.state_dict(), 'sdA2_2.pt')

## Testing network

# initialize figure
f, a = plt.subplots(3, N_VIEW_IMG-1, figsize=(8, 4))  # nrows,ncols,
plt.suptitle('Stacked Denoising Autoencoder (2 layers)', fontsize=16)
plt.ion()   # continuously plot

# original data (first row) for viewing

validation_data,npatch  = loadData(path_validation, patch_size,N_VIEW_IMG)
view_data = Variable(validation_data.view(-1, patch_size*patch_size))
view_noise_data = Variable(noise(view_data.data,MEAN,STDDEV))

encoded_data_1 = autoencoder_1.encoder(view_data)
encoded_data_2 = autoencoder_2.encoder(encoded_data_1)
decoded_data_2 = autoencoder_2.decoder(encoded_data_2)
decoded_data_1 = autoencoder_1.decoder(decoded_data_2)

view_data = np.reshape(view_data.data.numpy(), (-1,patch_size, patch_size))
view_noise_data = np.reshape(view_noise_data.data.numpy(), (-1,patch_size, patch_size))
decoded_data = np.reshape(decoded_data_1.data.numpy(), (-1,patch_size, patch_size))

for i in range(0,N_VIEW_IMG-1):
    reconstruction_view = reconstruct_from_patches_2d(view_data[npatch*(i):npatch*(i+1)], (height,width))
    reconstruction_view_noise = reconstruct_from_patches_2d(view_noise_data[npatch*(i):npatch*(i+1)], (height,width))
    

    a[0][i].imshow(reconstruction_view, cmap='gray')
    a[1][i].imshow(reconstruction_view_noise, cmap='gray')
    
    reconstruction_decoded = reconstruct_from_patches_2d(decoded_data[npatch*(i):npatch*(i+1)], (height,width))
    
    a[2][i].clear()
    a[2][i].imshow(reconstruction_decoded, cmap='gray')
    
    psnr    = compare_psnr(view_data[npatch*(i):npatch*(i+1)], decoded_data[npatch*(i):npatch*(i+1)])
    sim     = ssim(view_data[npatch*(i):npatch*(i+1)], decoded_data[npatch*(i):npatch*(i+1)])
    
    print('PSNR image ', i+1 , ': ', psnr)
    print('SSIM image ', i+1 , ': ', sim)
                                             
            


                    
plt.draw(); plt.pause(0.05)



plt.ioff()
plt.show()
