from time import time

import numpy as np

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

def learn_dictionary(img,n_components=100,alpha=1,plot=False,n_iter=500):
    
    height,width    = img.shape
    

    print('Extracting reference patches...')
    
    t0              = time()
    patch_size      = (7, 7)
    data            = extract_patches_2d(img, patch_size)
    data            = data.reshape(data.shape[0], -1)
    data            -= np.mean(data, axis=0)
    data            /= np.std(data, axis=0)
    
    print('done in %.2fs.' % (time() - t0))

    print('Learning the dictionary...')
    
    t0              = time()
    dico            = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
    V               = dico.fit(data).components_
    dt              = time() - t0
    
    print('done in %.2fs.' % dt)
    
    if plot == True:
        
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

    
    return dico, V
    
    
    

