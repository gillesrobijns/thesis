import sys
from skimage import io, color, restoration, img_as_float
from skimage.color import rgb2gray,gray2rgb
from scipy.signal import convolve2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
import numpy as np
from dA import dA

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

img                     = io.imread('original.png', as_grey=True)
img                     = rgb2gray(img_as_float(img))
img                     = np.array(img)



psf                     = np.ones((5, 5)) / 25
img                     += 0.001 * img.std() * np.random.standard_normal(img.shape)
print(img)
            
deconvolved_RL          = restoration.richardson_lucy(img, psf, iterations=30)




