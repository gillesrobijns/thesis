import matplotlib.pyplot as plt


def plot(nrows,ncols,data,title):
    
    # initialize figure
    f, a        = plt.subplots(nrows, ncols, figsize=(8, 4)) 
     
    plt.style.use("ggplot")
    plt.ion()   # continuously plot
    plt.suptitle(title, fontsize=16)
    
    
    for i in range(0,nrows): 
        
        for j in range(0,ncols):
                
            a[0][i].imshow(reconstruction_view, cmap='gray')
            a[0][i].axis('off')
            a[1][i].imshow(reconstruction_view_noise, cmap='gray')
            a[1][i].axis('off')
            
            reconstruction_decoded = reconstruct_from_patches_2d(decoded_data[npatch*(i):npatch*(i+1)], (height,width))
        
            a[2][i].imshow(reconstruction_decoded, cmap='gray')
            a[2][i].axis('off')
        
        psnr    = compare_psnr(view_data[npatch*(i):npatch*(i+1)], decoded_data[npatch*(i):npatch*(i+1)])
        sim     = ssim(view_data[npatch*(i):npatch*(i+1)], decoded_data[npatch*(i):npatch*(i+1)])
        
        print('PSNR image ', i+1 , ': ', psnr)
        print('SSIM image ', i+1 , ': ', sim)
        
                                                 
                
    
    
                        
    plt.draw(); plt.pause(0.05)
    
    
    
    plt.ioff()
    plt.show()