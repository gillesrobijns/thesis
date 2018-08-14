from time import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import skimage.data
from skimage.measure import compare_psnr
import matplotlib.pyplot as plt
from skimage import io

import pybm3d

from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

from os import listdir

path      = "/users/gillesrobijns/Documents/Thesis/images/Train/bastanievlayenchiesi1_2013-12-18_0003vbrok2_f401.png"
noise_std_dev   = 0.25

def noise(inp, mean=0, stddev=1):

    noise = np.random.normal(loc=mean, scale=stddev, size=np.shape(inp))

    out = np.add(inp, noise)

    return out


img = io.imread(path,as_grey=True)

noisy_img = noise(img,0,noise_std_dev).astype(img.dtype)


out = pybm3d.bm3d.bm3d(noisy_img, noise_std_dev)

noise_psnr = compare_psnr(img, noisy_img)
out_psnr = compare_psnr(img, out)

print("PSNR of noisy image: ", noise_psnr)
print("PSNR of reconstructed image: ", out_psnr)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 6),
                       sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})

ax[0].imshow(img,cmap='gray')
ax[0].axis('off')
ax[0].set_title('0riginal')
ax[1].imshow(noisy_img,cmap='gray')
ax[1].axis('off')
ax[1].set_title('Noisy')
ax[2].imshow(out,cmap='gray')
ax[2].axis('off')
ax[2].set_title('Denoised with bm3d')


fig.tight_layout()


plt.show()