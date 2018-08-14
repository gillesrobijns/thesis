from time import time

import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage import img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.color import rgb2gray

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

import torch
import torch.nn as nn

import pybm3d




filename = "/users/gillesrobijns/Documents/Thesis/images/Train/bastanievlayenchiesi1_2013-12-18_0003vbrok2_f401.png"
file = io.imread(filename)
img = rgb2gray(img_as_float(file))
height, width = img.shape

sigma = 0.2
noisy = img + sigma * np.random.standard_normal(img.shape)
noisy = np.clip(noisy, 0, 1)

print(noisy.shape)

# NL MEANS

print('Starting NL Means')

sigma_est = np.mean(estimate_sigma(noisy, multichannel=True))
print("estimated noise standard deviation = {}".format(sigma_est))

patch_kw = dict(patch_size=20,      # 5x5 patches
                patch_distance=21,  # 13x13 search area
                multichannel=True)

# slow algorithm
t0 = time()
denoise_slow = denoise_nl_means(noisy, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
dt = time() - t0
print('Slow algorithm done in %.2fs.' % dt)

# fast algorithm
t0 = time()
denoise_fast = denoise_nl_means(noisy, h=0.8 * sigma_est, fast_mode=True,
                                **patch_kw)
dt = time() - t0
print('Fast algorithm done in %.2fs.' % dt)

#DENOISING AUTOENCODER 
patch_size = 12
HUNITS = 10
path_save= "/users/gillesrobijns/Documents/Thesis/dA.pt"


#decoded = autoencoder(patch_size*patch_size,HUNITS)

# DICTIONARY
print('Starting Dictionary')
# Extract all reference patches from the left half of the image
print('Extracting reference patches...')
t0 = time()
patch_size = (7, 7)
data = extract_patches_2d(noisy, patch_size)
data = data.reshape(data.shape[0], -1)
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
print('done in %.2fs.' % (time() - t0))

# Learn the dictionary from reference patches
print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
V = dico.fit(data).components_
dt = time() - t0
print('done in %.2fs.' % dt)

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Dictionary learned from face patches\n' +
             'Train time %.1fs on %d patches' % (dt, len(data)),
             fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

# #############################################################################
# Extract noisy patches and reconstruct them using the dictionary

print('Extracting noisy patches... ')
t0 = time()
data = extract_patches_2d(noisy, patch_size)
data = data.reshape(data.shape[0], -1)
intercept = np.mean(data, axis=0)
data -= intercept
print('done in %.2fs.' % (time() - t0))



print('Orthogonal Matching Pursuit: 1 atom')
reconstructions_omp1 = img.copy()
t0 = time()
reconstructions = {}
dico.set_params(transform_algorithm = 'omp', **{'transform_n_nonzero_coefs': 1})
code = dico.transform(data)
patches = np.dot(code, V)
patches += intercept
patches = patches.reshape(len(data), *patch_size)
reconstructions_omp1 = reconstruct_from_patches_2d(
        patches, (height, width))
dt = time() - t0
print('done in %.2fs.' % dt)

print('Orthogonal Matching Pursuit: 2 atoms')
reconstructions_omp2 = img.copy()
t0 = time()
reconstructions = {}
dico.set_params(transform_algorithm = 'omp', **{'transform_n_nonzero_coefs': 2})
code = dico.transform(data)
patches = np.dot(code, V)
patches += intercept
patches = patches.reshape(len(data), *patch_size)
reconstructions_omp1 = reconstruct_from_patches_2d(
        patches, (height, width))
dt = time() - t0
print('done in %.2fs.' % dt)

print('BM3D')
t0 = time()
bm3d = pybm3d.bm3d.bm3d(noisy, sigma)
dt = time() - t0
print('done in %.2fs.' % dt)


fig, ax = plt.subplots(nrows=1, ncols=7, figsize=(8, 6),
                       sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})


ax[0].imshow(img,cmap='gray')
ax[0].axis('off')
ax[0].set_title('original\n(noise free)')
ax[1].imshow(noisy,cmap='gray')
ax[1].axis('off')
ax[1].set_title('Noisy Image')
ax[2].imshow(denoise_slow,cmap='gray')
ax[2].axis('off')
ax[2].set_title('non-local means\n(slow)')
ax[3].imshow(reconstructions_omp2,cmap='gray')
ax[3].axis('off')
ax[3].set_title('dictionary (omp2)')
ax[4].imshow(denoise_fast,cmap='gray')
ax[4].axis('off')
ax[4].set_title('non-local means\n(fast)')
ax[5].imshow(bm3d,cmap='gray')
ax[5].axis('off')
ax[5].set_title('BM3D')
ax[6].imshow(reconstructions_omp1,cmap='gray')
ax[6].axis('off')
ax[6].set_title('dictionary (omp1)')

fig.tight_layout()


plt.show()