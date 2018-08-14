from time import time
from os import listdir
from skimage import io
from skimage import img_as_float
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import reconstruct_from_patches_2d,extract_patches_2d
from dA import dA
from NLMeans import NLMeans
import pybm3d

path = "/users/gillesrobijns/Documents/Thesis/Images/Test/"

def loadData(path,nImages,height,width):

    print('Loading images from' + path +'...')
    
    t0 = time()

    imagesList      = listdir(path)
    loadedImages    = np.array([])
    
    for i in range(0,nImages+1):
        if not imagesList[i] == ".DS_Store":
                
            img             = io.imread(path + imagesList[i],as_grey=True)
            loadedImages    = np.append(loadedImages, img[0:height,0:width])
                
    
    out             = np.reshape(loadedImages,(-1,height,width))
    
    print('Images loaded in %.2fs.' % (time() - t0))
    
    return out

def denoise_dA(img,HEIGHT,WIDTH,PATCH_SIZE,HUNITS):
    
    pat                     = extract_patches_2d(img[0:HEIGHT,0:WIDTH], (PATCH_SIZE,PATCH_SIZE))
    img_patches             = np.reshape(pat,(-1,PATCH_SIZE,PATCH_SIZE))
        
    _,denoised_dA           = dA(img_patches,PATCH_SIZE*PATCH_SIZE,HUNITS,HEIGHT,WIDTH)

    denoised_dA             = np.reshape(denoised_dA.data.numpy(), (-1,PATCH_SIZE, PATCH_SIZE))
    denoised_dA             = reconstruct_from_patches_2d(denoised_dA, (HEIGHT,WIDTH))
    
    return denoised_dA


images              = loadData(path,10,510,510)

for i in range(0,10):

    
    DA          = denoise_dA(images[i], 510, 510, 12, 500)
    NL          = NLMeans(images[i],12,11,0.08)
    BM          = pybm3d.bm3d.bm3d(images[i], 0.15)
    
    DA_save     = '/users/gillesrobijns/Documents/Thesis/Denoised_images/da/' + 'da_' + str(i) +'.png'
    NL_save     = '/users/gillesrobijns/Documents/Thesis/Denoised_images/nl/' + 'nl_' + str(i) +'.png'
    BM_save     = '/users/gillesrobijns/Documents/Thesis/Denoised_images/bm/' + 'bm_' + str(i) +'.png'
    
    io.imsave(DA_save, DA)
    io.imsave(NL_save, NL)
    io.imsave(BM_save, BM)
    
    

print('Batch Processing Done')
    



