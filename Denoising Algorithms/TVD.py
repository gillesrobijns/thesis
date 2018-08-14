import numpy as np
import matplotlib.pyplot as plt

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float, color, io
from skimage.util import random_noise

from os import listdir



path = "/users/gillesrobijns/Documents/Thesis/images/"

def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList[1:]:
        if not image == ".DS_Store":
            img = io.imread(path + image)
            loadedImages.append(img)

    return loadedImages


# your images in an array
imgs = loadImages(path)


sigma = 0.25

def noise(image):
    original = []
    noisy = []
    for i in range(4):
        original.append(img_as_float(image[i]))
        noisy.append(random_noise(original[i], var=sigma**2))
        
    return noisy

noisy = noise(imgs)

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 5), sharex=True,
                       sharey=True, subplot_kw={'adjustable': 'box-forced'})

plt.gray()



ax[0, 0].imshow(noisy[0])
ax[0, 0].axis('off')
ax[0, 0].set_title('Image 1')
ax[0, 1].imshow(noisy[1])
ax[0, 1].axis('off')
ax[0, 1].set_title('Image 2')
ax[0, 2].imshow(noisy[2])
ax[0, 2].axis('off')
ax[0, 2].set_title('Image 3')
ax[0, 3].imshow(noisy[3])
ax[0, 3].axis('off')
ax[0, 3].set_title('Image 4')

ax[1, 0].imshow(denoise_tv_chambolle(noisy[0], weight=0.3, multichannel=True))
ax[1, 0].axis('off')
ax[1, 1].imshow(denoise_tv_chambolle(noisy[1], weight=0.3, multichannel=True))
ax[1, 1].axis('off')
ax[1, 2].imshow(denoise_tv_chambolle(noisy[2], weight=0.3, multichannel=True))
ax[1, 2].axis('off')
ax[1, 3].imshow(denoise_tv_chambolle(noisy[3], weight=0.3, multichannel=True))
ax[1, 3].axis('off')


fig.tight_layout()

plt.show()