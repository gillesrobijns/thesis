from time import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from skimage import color, restoration
from loadData import loadData
from noise import noise
from torch.autograd import Variable
import torch.nn as nn
from sdA import sdA2,sdA3
from skimage import io,filters,feature
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim as ssim


path_train          = "/users/gillesrobijns/Documents/Thesis/Images/Train/"
path_validation     = "/users/gillesrobijns/Documents/Thesis/Images/Train/"

PATCH_SIZE          = 12
N_TEST_IMG          = 50
N_VIEW_IMG          = 5
MEAN                = 0
STDDEV              = 0.4
HUNITS              = 100
HUNITS1             = 200
HUNITS2             = 150
HUNITS3             = 150
HEIGHT              = 510
WIDTH               = 510


test_data_full, _           = loadData(path_validation,False,PATCH_SIZE,N_TEST_IMG,HEIGHT,WIDTH)
test_data_full_noise        = noise(test_data_full,MEAN,STDDEV)


print('Starting Stacked Denoising Autoencoder (2 layers)')

f, a = plt.subplots(3, N_VIEW_IMG, figsize=(8, 4))  # nrows,ncols,
plt.style.use("ggplot")
plt.ion()   # continuously plot
plt.suptitle('sdA2', fontsize=16)
print('Plotting images...')

times   = np.array([])
psnrs   = np.array([])
sims    = np.array([])

for i in range(N_TEST_IMG):
    
    t0                      = time()
    
    noisy_pat               = extract_patches_2d(test_data_full_noise[i], (PATCH_SIZE,PATCH_SIZE))
    _,denoised_sdA2         = sdA2(noisy_pat,PATCH_SIZE*PATCH_SIZE,HUNITS1,HUNITS2)

    denoised_sdA2           = np.reshape(denoised_sdA2.data.numpy(), (-1,PATCH_SIZE, PATCH_SIZE))
    denoised_sdA2           = reconstruct_from_patches_2d(denoised_sdA2, (HEIGHT,WIDTH))
    
    dt                      = time() - t0
    
    psnr                    = compare_psnr(test_data_full[i],denoised_sdA2)
    sim                     = ssim(test_data_full[i],denoised_sdA2)
    
    psnrs                   = np.append(psnr,psnrs)
    times                   = np.append(times, dt)
    sims                    = np.append(sim,sims)
    
    if i < N_VIEW_IMG:

        a[0][i].imshow(test_data_full[i], cmap='gray')
        a[0][i].axis('off')
        a[1][i].imshow(test_data_full_noise[i], cmap='gray')
        a[1][i].axis('off')
        a[2][i].imshow(denoised_sdA2, cmap='gray')
        a[2][i].axis('off')

    print(' PSNR ' ,  psnr , 'SSIM', sim , 'Time ',dt , )
    
psnr_mean       = np.mean(psnrs)  
psnr_std        = np.std(psnrs)
times_mean      = np.mean(times) 
times_std       = np.std(times) 
sim_mean        = np.mean(sims) 
sim_std         = np.std(sims) 

print('PSNR: ', psnr_mean, '+-', psnr_std , 'SSIM: ', sim_mean, '+-', sim_std, ' Time: ' ,times_mean, '+-' ,times_std)

plt.draw(); plt.pause(0.05)
plt.ioff()
plt.show()
    