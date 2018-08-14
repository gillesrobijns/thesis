from time import time
import numpy as np
from os import listdir
from skimage import io
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from skimage.color import rgb2gray
from skimage import img_as_float



def loadData(path,patches,patchSize,nImages,height,width):

    print('Loading images from' + path +'...')
    
    t0 = time()

    imagesList      = listdir(path)
    loadedImages    = np.array([])
    
    if patches == True:
        
        for i in range(0,nImages+1):
        
            if not imagesList[i] == ".DS_Store":
                img             = io.imread(path + imagesList[i],as_grey=True)
                img             = rgb2gray(img_as_float(img))
                pat             = extract_patches_2d(img[0:height,0:width], (patchSize,patchSize))
                loadedImages    = np.append(loadedImages, pat)
                
        npatches        = pat.shape[0]
        out             = np.reshape(loadedImages,(-1,patchSize,patchSize))

        
        
    else:
        
        for i in range(0,nImages+1):
            if not imagesList[i] == ".DS_Store":
                
                img             = io.imread(path + imagesList[i],as_grey=True)
                loadedImages    = np.append(loadedImages, img[0:height,0:width])
                
        npatches        = 1
        out             = np.reshape(loadedImages,(-1,height,width))
        
    
    print('Images loaded in %.2fs.' % (time() - t0))
    
    return out, npatches
