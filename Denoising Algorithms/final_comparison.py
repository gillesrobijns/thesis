from time import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage import color, restoration
from scipy.signal import convolve2d as conv2
from loadData import loadData
from noise import noise
from NLMeans import NLMeans
import pybm3d
from dict import learn_dictionary
from torch.autograd import Variable
import torch.nn as nn
from dA import dA
from sdA import sdA2,sdA3
from VAE import VAE
from DVAE import DVAE
from skimage import io,filters,feature


path_train          = "/users/gillesrobijns/Documents/Thesis/Images/Train/"
path_validation     = "/users/gillesrobijns/Documents/Thesis/images/Train/bastanievlayenchiesi1_2013-12-18_0003vbrok2_f224.png"
path_dict           = "/users/gillesrobijns/Documents/Thesis/images/Train/bastanievlayenchiesi1_2013-12-18_0003vbrok2_f224.png"

img_validation      = io.imread(path_validation,as_grey=True)
img_dict            = io.imread(path_dict,as_grey=True)

PATCH_SIZE          = 12
N_TEST_IMG          = 1
MEAN                = 0
STDDEV              = 0.2
HUNITS              = 100
HUNITS1             = 200
HUNITS2             = 150
HUNITS3             = 100
HEIGHT              = 510
WIDTH               = 510
patch_size          = (7, 7)



test_data_full_noise        = noise(img_validation[0:HEIGHT,0:WIDTH],MEAN,STDDEV)

noisy_pat                   = extract_patches_2d(test_data_full_noise, (PATCH_SIZE,PATCH_SIZE))


###################################  WAVELET #################################################################

print('Wavelet Denoising')

denoised_wavelet                        = denoise_wavelet(test_data_full_noise, sigma=0.2,wavelet='db1')

###################################  DICTIONARY ##############################################################

print('Starting Dictionary')

dico, V  = learn_dictionary(img_dict[0:HEIGHT,0:WIDTH],alpha=1,n_components=100,n_iter=500)

print('Extracting noisy patches... ')
        
t0              = time()
data            = extract_patches_2d(test_data_full_noise, patch_size)
data            = data.reshape(data.shape[0], -1)
intercept       = np.mean(data, axis=0)
data            -= intercept
        
print('done in %.2fs.' % (time() - t0))
    
    
print('Orthogonal Matching Pursuit: 2 atoms')
        
t0                          = time()
        
dico.set_params(transform_algorithm = 'omp', **{'transform_n_nonzero_coefs': 2})
        
code                        = dico.transform(data)
patches                     = np.dot(code, V)
patches                     += intercept
patches                     = patches.reshape(len(data), *patch_size)
reconstructions_omp2        = reconstruct_from_patches_2d(patches, (HEIGHT, WIDTH))
dt                          = time() - t0
        
print('done in %.2fs.' % dt)

###################################  NON-LOCAL MEANS #########################################################

print('Starting NL Means')

denoised_nlmeans       = NLMeans(test_data_full_noise,7,11,0.2)

###################################  BM3D ####################################################################

print('Starting BM3D')

t0                      = time()
denoised_bm3d           = pybm3d.bm3d.bm3d(test_data_full_noise, 0.15)
dt                      = time() - t0

print('BM3D done in %.2fs.' % dt)

###################################  DENOISING AUTOENCODERS ##################################################

print('Starting Denoising Autoencoder')

_,denoised_dA           = dA(noisy_pat,PATCH_SIZE*PATCH_SIZE,HUNITS,HEIGHT,WIDTH)

denoised_dA             = np.reshape(denoised_dA.data.numpy(), (-1,PATCH_SIZE, PATCH_SIZE))
denoised_dA             = reconstruct_from_patches_2d(denoised_dA, (HEIGHT,WIDTH))

print('Starting Stacked Denoising Autoencoder (2 layers)')

_,denoised_sdA2         = sdA2(noisy_pat,PATCH_SIZE*PATCH_SIZE,HUNITS1,HUNITS2)

denoised_sdA2           = np.reshape(denoised_sdA2.data.numpy(), (-1,PATCH_SIZE, PATCH_SIZE))
denoised_sdA2           = reconstruct_from_patches_2d(denoised_sdA2, (HEIGHT,WIDTH))

print('Starting Stacked Denoising Autoencoder (3 layers)')

_,denoised_sdA3         = sdA3(noisy_pat,PATCH_SIZE*PATCH_SIZE,HUNITS1,HUNITS2,HUNITS3)

denoised_sdA3           = np.reshape(denoised_sdA3.data.numpy(), (-1,PATCH_SIZE, PATCH_SIZE))
denoised_sdA3           = reconstruct_from_patches_2d(denoised_sdA3, (HEIGHT,WIDTH))
      

###################################  VARIATIONAL AUTOENCODER #################################################

print('Starting Variational Autoencoder')

denoised_VAE            = VAE(noisy_pat,PATCH_SIZE*PATCH_SIZE,100,30,30)

denoised_VAE            = np.reshape(denoised_VAE.data.numpy(), (-1,PATCH_SIZE, PATCH_SIZE))
denoised_VAE            = reconstruct_from_patches_2d(denoised_VAE, (HEIGHT,WIDTH))

###################################  DENOISING VARIATIONAL AUTOENCODER #######################################

print('Starting Denoising Variational Autoencoder')

denoised_DVAE           = DVAE(noisy_pat,PATCH_SIZE*PATCH_SIZE,200,50,50)

denoised_DVAE           = np.reshape(denoised_DVAE.data.numpy(), (-1,PATCH_SIZE, PATCH_SIZE))
denoised_DVAE           = reconstruct_from_patches_2d(denoised_DVAE, (HEIGHT,WIDTH))



###################################  PLOTTINGS ###############################################################

# initialize figure
f, a = plt.subplots(4, 3, figsize=(8, 4))  # nrows,ncols,
plt.style.use("ggplot")
plt.ion()   # continuously plot
plt.suptitle('Comparison between denoising techniques', fontsize=16)

print('Plotting images...')


a[0][0].imshow(img_validation, cmap='gray')
a[0][0].axis('off')
a[0][0].set_title('Original')

a[1][0].imshow(test_data_full_noise, cmap='gray')
a[1][0].axis('off')
a[1][0].set_title('Noisy') 
   
a[2][0].imshow(reconstructions_omp2, cmap='gray')
a[2][0].axis('off')
a[2][0].set_title('Dictonary') 

a[3][0].imshow(denoised_DVAE, cmap='gray')
a[3][0].axis('off')
a[3][0].set_title('DVAE')

a[0][1].imshow(denoised_nlmeans, cmap='gray')
a[0][1].axis('off')
a[0][1].set_title('NL Means')

a[1][1].imshow(denoised_bm3d, cmap='gray')
a[1][1].axis('off')
a[1][1].set_title('BM3D') 

a[2][1].imshow(denoised_dA, cmap='gray')
a[2][1].axis('off')
a[2][1].set_title('DA') 

a[0][2].imshow(denoised_sdA2, cmap='gray')
a[0][2].axis('off')
a[0][2].set_title('sDA (2)')

a[1][2].imshow(denoised_sdA3, cmap='gray')
a[1][2].axis('off')
a[1][2].set_title('sDA (3)')

a[2][2].imshow(denoised_VAE, cmap='gray')
a[2][2].axis('off')
a[2][2].set_title('VAE')  

a[3][2].imshow(denoised_wavelet, cmap='gray')
a[3][2].axis('off')
a[3][2].set_title('Wavelet')  
                                   
            

                    
plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()