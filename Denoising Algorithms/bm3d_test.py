from time import time

import numpy as np
import matplotlib.pyplot as plt
from skimage import color, restoration
from loadData import loadData
from noise import noise
import pybm3d
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


print('Starting BM3D')

times   = np.array([])
psnrs   = np.array([])
sims    = np.array([])

f, a = plt.subplots(3, N_VIEW_IMG, figsize=(8, 4))  # nrows,ncols,
plt.style.use("ggplot")
plt.ion()   # continuously plot
plt.suptitle('BM3D', fontsize=16)
print('Plotting images...')
    

for i in range(N_TEST_IMG):
    
    t0                      = time()

    denoised_bm3d           = pybm3d.bm3d.bm3d(test_data_full_noise[i], 0.4)
    
    dt                      = time() - t0
    
    psnr                    = compare_psnr(test_data_full[i],denoised_bm3d)
    sim                     = ssim(test_data_full[i],denoised_bm3d)
    
    psnrs                   = np.append(psnr,psnrs)
    times                   = np.append(times, dt)
    sims                    = np.append(sim,sims)

    if i < N_VIEW_IMG:

        a[0][i].imshow(test_data_full[i], cmap='gray')
        a[0][i].axis('off')
        a[1][i].imshow(test_data_full_noise[i], cmap='gray')
        a[1][i].axis('off')
        a[2][i].imshow(denoised_bm3d, cmap='gray')
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