import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


n_groups    = 9

means_psnr  = (27.09798873,22.17421919,16.81095082,21.63000251,20.1138216,20.76327678,20.95582548,22.09639587,21.336295227)
std_psnr    = (0.990865771,0.57237278,1.259335098,0.257090872,0.340187591,0.175410494,0.263834041,0.194183091,0.192841818)

means_time  = (2.523826928,4.508795376,5.814128828,3.276897993,4.923395772,0.019361973,38.22149227,7.793437185,2.168449006)
std_time    = (0.117624808,0.251073927,0.649919497,0.200017406,0.358263237,0.007486078,2.429753063,0.394298765,0.02166786)

fig, ax1    = plt.subplots()
plt.style.use("ggplot")

index       = np.arange(n_groups)
bar_width   = 0.35

opacity = 0.4


rects1 = ax1.bar(index, means_psnr, bar_width,
                alpha=opacity, color='b',
                yerr=std_psnr,
                label='psnr')

ax1.set_ylabel('PSNR (dB)',color='b')
ax1.set_title('Comparison of denoising methods')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(('dA', 'sdA2', 'sdA3', 'VAE', 'DVAE', 'Wav', 'Dict', 'BM3D', 'NLMeans'))

ax2 = ax1.twinx()
ax2.set_ylabel('Time (s)',color='r')
rects2 = ax2.bar(index + bar_width, means_time, bar_width,
                alpha=opacity, color='r',
                yerr=std_time,
                label='time')



fig.tight_layout()
plt.show()