from time import time
import numpy as np

from skimage import io
from skimage import img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.color import rgb2gray


def NLMeans(img,patch_size,patch_distance,h):
    
    sigma_est   = np.mean(estimate_sigma(img, multichannel=False))
    t0          = time()
    denoised    = denoise_nl_means(img,patch_size,patch_distance, h,multichannel=False )
    dt          = time() - t0
    
    print('NLMeans done in %.2fs.' % dt)
    
    return denoised

