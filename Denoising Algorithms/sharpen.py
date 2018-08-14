from time import time
from os import listdir
from skimage import color, restoration
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage import io
from skimage import img_as_float
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import reconstruct_from_patches_2d,extract_patches_2d
from dA import dA

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

def sharpen_deconvolution(img):
    
    psf                 = np.ones((7, 7)) / 100
    deconvolved_RL      = restoration.richardson_lucy(img, psf, iterations=30)         

    return deconvolved_RL

def sharpen_unsharpmask(img,radius,amount):
    
    blurred             = gaussian_filter(img,sigma=radius,mode='reflect')
    result              = img + (img - blurred) * amount
    
    return result


images              = loadData(path,3,510,510)
images              = np.append(images,np.zeros((510,510)))
images              = np.reshape(images,(-1,510,510))
images[3,250,250]   = 1


f, a = plt.subplots(3, 4, figsize=(8, 4))  # nrows,ncols,
plt.style.use("ggplot")
plt.ion()   # continuously plot
plt.suptitle('Sharpening', fontsize=16)

print('Plotting images...')

for i in range(0,4):
    
    images[i] = denoise_dA(images[i], 510, 510, 12,500)
    
    a[0][i].imshow(images[i], cmap='gray')
    a[0][i].axis('off')
    a[0][i].set_title('Denoised')

    a[1][i].imshow(sharpen_deconvolution(images[i]), cmap='gray')
    a[1][i].axis('off')
    a[1][i].set_title('Deconvolution')

    a[2][i].imshow(sharpen_unsharpmask(images[i],3,0.75), cmap='gray')
    a[2][i].axis('off')
    a[2][i].set_title('Unsharp Masking')
    

plt.draw(); plt.pause(0.05)
plt.ioff()
plt.show()
    
    
    
    

