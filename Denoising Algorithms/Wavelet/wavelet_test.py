from skimage.restoration import denoise_wavelet, estimate_sigma
import numpy as np
import matplotlib.pyplot as plt
from loadData import loadData
from skimage import io
from skimage.measure import compare_psnr
from time import time

path_train          = "/users/gillesrobijns/Documents/Thesis/Images/Train/"
path_validation     = "/users/gillesrobijns/Documents/Thesis/Images/Test/"

PATCH_SIZE          = 12
N_TEST_IMG          = 1
HEIGHT              = 510
WIDTH               = 510

path      = "/users/gillesrobijns/Documents/Thesis/images/Train/bastanievlayenchiesi1_2013-12-18_0003vbrok2_f401.png"
noise_std_dev   = 0.06


def noise(inp, mean=0, stddev=1):

    noise = np.random.normal(loc=mean, scale=stddev, size=np.shape(inp))

    out = np.add(inp, noise)

    return out


img = io.imread(path,as_grey=True)

noisy_img = noise(img,0,noise_std_dev).astype(img.dtype)

t0                      = time()
denoised_wavelet        = denoise_wavelet(noisy_img, sigma=0.1, wavelet='db1')
dt                      = time() - t0
print('Wavelet thresholding done in %.2fs.' % dt)

noise_psnr = compare_psnr(img, noisy_img)
out_psnr = compare_psnr(img, denoised_wavelet.astype(img.dtype))

print("PSNR of noisy image: ", noise_psnr)
print("PSNR of reconstructed image: ", out_psnr)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 6),
                       sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})

ax[0].imshow(img,cmap='gray')
ax[0].axis('off')
ax[0].set_title('Original')
ax[1].imshow(noisy_img,cmap='gray')
ax[1].axis('off')
ax[1].set_title('Noisy')
ax[2].imshow(denoised_wavelet,cmap='gray')
ax[2].axis('off')
ax[2].set_title('Denoised with Wavelet thresholding')


fig.tight_layout()


plt.show()
