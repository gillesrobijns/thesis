from time import time
from dict import learn_dictionary
import numpy as np
import matplotlib.pyplot as plt
from loadData import loadData
from skimage import io
from skimage.measure import compare_psnr
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from loadData import loadData
from noise import noise
from skimage.measure import compare_ssim as ssim


path_train                  = "/users/gillesrobijns/Documents/Thesis/images/Test/bastanie_2012-07-11_0002vblok_shadowremoved_f045.png"
path_validation             = "/users/gillesrobijns/Documents/Thesis/Images/Train/"
PATCH_SIZE                  = 12
N_TEST_IMG                  = 20
N_VIEW_IMG                  = 5
MEAN                        = 0
STDDEV                      = 0.4
HEIGHT                      = 510
WIDTH                       = 510
patch_size                  = (7, 7)
test_data_full, _           = loadData(path_validation,False,PATCH_SIZE,N_TEST_IMG,HEIGHT,WIDTH)
test_data_full_noise        = noise(test_data_full,MEAN,STDDEV)


img_train   = io.imread(path_train,as_grey=True)


print('Starting Dictionary')

dico, V  = learn_dictionary(img_train[0:HEIGHT,0:WIDTH],alpha=1,n_components=100,n_iter=500)

times   = np.array([])
psnrs   = np.array([])
sims    = np.array([])

f, a = plt.subplots(3, N_VIEW_IMG, figsize=(8, 4))  # nrows,ncols,
plt.style.use("ggplot")
plt.ion()   # continuously plot
plt.suptitle('Onilne dictionary learning', fontsize=16)
print('Plotting images...')

for i in range(N_TEST_IMG):

    print('Extracting noisy patches... ')
        
    t0              = time()
    data            = extract_patches_2d(test_data_full_noise[i], patch_size)
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
    
    psnr                    = compare_psnr(test_data_full[i],reconstructions_omp2)
    sim                     = ssim(test_data_full[i],reconstructions_omp2)
    
    psnrs                   = np.append(psnr,psnrs)
    times                   = np.append(times, dt)
    sims                    = np.append(sim,sims)
    
    print('Time ',dt ,' PSNR ' ,  psnr)
    
    if i < N_VIEW_IMG:

        a[0][i].imshow(test_data_full[i], cmap='gray')
        a[0][i].axis('off')
        a[1][i].imshow(test_data_full_noise[i], cmap='gray')
        a[1][i].axis('off')
        a[2][i].imshow(reconstructions_omp2, cmap='gray')
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