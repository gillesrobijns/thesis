import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t       = np.arange(0.1, 0.7, 0.1)
psnr    = np.array([22.1276,24.49,24.97,22.8899,20.3456,18.2072]) 
ssim    = np.array([0.5975,0.6512,0.7459,0.6541,0.5129,0.4372])


# Note that using plt.subplots below is equivalent to using
# fig = plt.figure() and then ax = fig.add_subplot(111)
fig, ax1 = plt.subplots()
ax1.plot(t, psnr)
plt.style.use("ggplot")

ax1.set(xlabel='Corruption level')

ax1.set_ylabel('PSNR (dB)',color='b')
ax1.set_title('Influence of corruption level on dA')

ax2 = ax1.twinx()
ax2.set_ylabel('SSIM',color='r')
ax2.plot(t, ssim)

fig.savefig("test.png")
plt.show()