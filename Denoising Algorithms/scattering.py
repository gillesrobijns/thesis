from time import time
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from skimage.color import rgb2gray
from skimage import img_as_float
from skimage import io
from os import listdir
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim as ssim

path_train          = "/users/gillesrobijns/Documents/Thesis/Images/Train/"
path_validation     = "/users/gillesrobijns/Documents/Thesis/Images/Test/"


EPOCH               = 50
BATCH_SIZE          = 4
LR                  = 0.0001         
N_TRAIN_IMG         = 350
N_VIEW_IMG          = 5
HEIGHT              = 200
WIDTH               = 200

def loadData(path,height,width,nImages):
    # return array of images
    print('Loading images from' + path_train +'...')
    t0 = time()

    imagesList      = listdir(path)
    loadedImages    = np.array([])
    
    for i in range(0,nImages):
        
        if not imagesList[i] == ".DS_Store":
            img             = io.imread(path + imagesList[i],as_grey=True)
            img             = resize(img,(height,width))
            loadedImages    = np.append(loadedImages, img)
    
             
    loadedImages   = np.reshape(loadedImages,(-1,1,height,width)) 
    loadedImages   = torch.from_numpy(loadedImages).float()
    
    print('Images loaded in %.2fs.' % (time() - t0))
    
    return loadedImages

train_data          = loadData(path_train,HEIGHT,WIDTH,N_TRAIN_IMG)
train_loader        = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class ScatterAutoencoder(nn.Module):
    def __init__(self):
        super(ScatterAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # b, 32, 100, 100
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 32, 50, 50
            nn.Conv2d(32, 64, 2, stride=2, padding=1),  # b, 64, 26, 26
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 64, 13, 13
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # b, 128, 6, 6
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)  # b, 128, 3, 3
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2),  # b, 64, 8, 8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2),  # b, 32, 18, 18
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 8, stride=4),  # b, 16, 76, 76
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, stride=4, padding=52),  # b, 16, 200, 200
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


SAE = ScatterAutoencoder()
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(SAE.parameters(), lr=LR,
                             weight_decay=1e-5)


loss_vector = np.array([])

for epoch in range(EPOCH):
    for step, x in enumerate(train_loader):

        b_x = Variable(x)   # batch x, shape (batch, 28*28)
        b_y = Variable(x)   # batch y, shape (batch, 28*28)  
        

        decoded = SAE(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(b_x), len(train_loader.dataset),
                100. * step / len(train_loader),
                loss.data[0] ))

torch.save(SAE.state_dict(), 'SAE.pt')

## Testing network

# initialize figure
f, a = plt.subplots(2, N_VIEW_IMG-1, figsize=(8, 4))  # nrows,ncols,
plt.style.use("ggplot")
plt.ion()   # continuously plot
plt.suptitle('Scattering Network', fontsize=16)


# original data (first row) for viewing

validation_data             = loadData(path_validation,HEIGHT,WIDTH,N_VIEW_IMG)
view_data                   = Variable(validation_data.view(-1,1, HEIGHT,WIDTH))

decoded_data = SAE(view_data)

view_data                   = np.reshape(view_data.data.numpy(), (-1,HEIGHT, WIDTH))
decoded_data                = np.reshape(decoded_data.data.numpy(), (-1,HEIGHT, WIDTH))

for i in range(0,N_VIEW_IMG-1):
    
    
    a[0][i].imshow(view_data[i], cmap='gray')
    a[0][i].axis('off')
    a[1][i].imshow(decoded_data[i], cmap='gray')
    a[1][i].axis('off')
    
    psnr    = compare_psnr(view_data[i], decoded_data[i])
    sim     = ssim(view_data[i], decoded_data[i])
    
    print('PSNR image ', i+1 , ': ', psnr)
    print('SSIM image ', i+1 , ': ', sim)
    
                                             
            


                    
plt.draw(); plt.pause(0.05)



plt.ioff()
plt.show()