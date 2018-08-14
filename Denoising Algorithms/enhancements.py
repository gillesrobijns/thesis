from time import time

import numpy as np
import matplotlib.pyplot as plt
from skimage import io,filters,feature
import torch
from skimage import color, restoration
from torch.autograd import Variable
import torch.nn as nn
from loadData import loadData
from skimage import exposure,measure
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.feature import blob_dog, blob_log, blob_doh
import cv2
from sdA import sdA2,sdA3
from VAE import VAE

from dA import dA

path_validation     = "/users/gillesrobijns/Documents/Thesis/Images/Test/"

PATCH_SIZE          = 12
N_TEST_IMG          = 1
MEAN                = 0
STDDEV              = 0.25
HUNITS              = 500
HUNITS1             = 200
HUNITS2             = 150
HUNITS3             = 150
HEIGHT              = 510
WIDTH               = 510

test_data_full, _           = loadData(path_validation,False,PATCH_SIZE,N_TEST_IMG,HEIGHT,WIDTH)
test_data_patches, npatch   = loadData(path_validation,True,PATCH_SIZE,N_TEST_IMG,HEIGHT,WIDTH)

print('Starting Denoising Autoencoder')

denoised_sdA2           = VAE(test_data_patches,PATCH_SIZE*PATCH_SIZE,100,30,30)

denoised_sdA2             = np.reshape(denoised_sdA2.data.numpy(), (-1,PATCH_SIZE, PATCH_SIZE))
denoised_sdA2             = reconstruct_from_patches_2d(denoised_sdA2, (HEIGHT,WIDTH))


edges                   = feature.canny(denoised_sdA2, sigma=2)
filtered_dA             = np.clip(denoised_sdA2 + edges,0,1)
dA_eq                   = exposure.equalize_hist(denoised_sdA2)
dA_adapteq              = exposure.equalize_adapthist(denoised_sdA2, clip_limit=0.01)
dA_sigmoid              = exposure.adjust_sigmoid(denoised_sdA2, cutoff=0.35, gain=10, inv=False)
contours                = measure.find_contours(dA_sigmoid,0.4)
blobs_dog               = blob_dog(denoised_sdA2, max_sigma=30, threshold=.8)
blobs_log               = blob_log(denoised_sdA2, min_sigma =8, max_sigma=30, num_sigma=10, threshold=.25)


psf = np.ones((13, 13)) / 25
deconvolved_RL = restoration.richardson_lucy(denoised_sdA2, psf, iterations=30)

f, a = plt.subplots(3, 3, figsize=(8, 4))  # nrows,ncols,
plt.style.use("ggplot")
plt.ion()   # continuously plot
plt.suptitle('Enhancements for denoising autoencoders', fontsize=16)

print('Plotting images...')


a[0][0].imshow(test_data_full[0], cmap='gray')
a[0][0].axis('off')
a[0][0].set_title('Original')

a[1][0].imshow(denoised_sdA2, cmap='gray')
a[1][0].axis('off')
a[1][0].set_title('DA') 
   
a[2][0].imshow(filtered_dA, cmap='gray')
a[2][0].axis('off')
a[2][0].set_title('DA + edge detection') 

a[0][1].imshow(dA_adapteq, cmap='gray')
a[0][1].axis('off')
a[0][1].set_title('OpenCv')

a[1][1].imshow(deconvolved_RL, cmap='gray')
a[1][1].axis('off')
a[1][1].set_title('Deconvolved') 

a[2][1].imshow(dA_sigmoid, cmap='gray')   
a[2][1].axis('off')
a[2][1].set_title('DA sigmoid') 

a[0][2].imshow(dA_sigmoid, cmap='gray')
a[0][2].axis('off')
a[0][2].set_title('DA + contours')

for contour in contours:
    a[0][2].plot(contour[:, 1], contour[:, 0], linewidth=1)
       
a[1][2].imshow(dA_sigmoid, cmap='gray')
a[1][2].axis('off')
a[1][2].set_title('DA sigmoid + blobs DOG')       
            
for blob in blobs_dog:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='lime', linewidth=1, fill=False)
        a[1][2].add_patch(c)
        
a[2][2].imshow(dA_sigmoid, cmap='gray')
a[2][2].axis('off')
a[2][2].set_title('DA sigmoid + blobs LOG')       
            
for blob in blobs_log:
        y, x, r = blob
        c = plt.Circle((x, y), r*2, color='red', linewidth=1, fill=False)
        a[2][2].add_patch(c)

                    
plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()