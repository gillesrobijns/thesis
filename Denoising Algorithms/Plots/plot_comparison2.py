import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
sigma           = np.array([ 0.1        , 0.2       , 0.25      , 0.3       ,  0.4      ])

psnr_da         = np.array([ 24.361     , 25.280    , 26.408    , 27.073    , 23.906    ]) 
psnr_sda2       = np.array([ 23.372     , 23.588    , 22.174    , 20.344    , 17.567    ]) 
psnr_sda3       = np.array([ 16.160     , 16.307    , 16.811    , 16.290    , 16.218    ]) 
psnr_vae        = np.array([ 23.511     , 24.465    , 24.527    , 23.753    , 20.967    ]) 
psnr_dvae       = np.array([ 23.806     , 24.669    , 24.641    , 23.859    , 21.264    ]) 
psnr_bm3d       = np.array([ 29.529     , 23.975    , 22.096    , 20.386    , 17.787    ]) 
psnr_nlmeans    = np.array([ 28.178     , 23.002    , 21.299    , 19.417    , 16.911    ]) 
psnr_dict       = np.array([ 28.618     , 22.934    , 20.956    , 19.221    , 16.754    ]) 
psnr_wav        = np.array([ 27.072     , 22.232    , 20.763    , 18.889    , 16.628    ]) 





# Note that using plt.subplots below is equivalent to using
# fig = plt.figure() and then ax = fig.add_subplot(111)
fig, ax = plt.subplots()


ax.plot(sigma, psnr_da)
ax.plot(sigma, psnr_sda2)
ax.plot(sigma, psnr_sda3)
ax.plot(sigma, psnr_vae)
ax.plot(sigma, psnr_dvae)
ax.plot(sigma, psnr_bm3d)
ax.plot(sigma, psnr_nlmeans)
ax.plot(sigma, psnr_dict)
ax.plot(sigma, psnr_wav)


plt.style.use("ggplot")

ax.set(xlabel='Corruption level')

ax.set_ylabel('PSNR (dB)')
ax.set_title('PSNR vs Sigma')

plt.legend(['dA','sdA2','sdA3','VAE','DVAE','BM3D','NlMeans','Dict','Wavelet'])

fig.savefig("test.png")
plt.show()