import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

def f(t,gain,cutoff):
    return 1/(1+np.exp(gain*(cutoff-t)))

t1 = np.arange(0.0, 1.0, 0.02)
t2 = np.arange(0.0, 1.0, 0.02)

plt.figure(1)
plt.subplot(121)
plt.plot(t2, f(t2,8,0.5), 'g', label='Gain = 8')
plt.plot(t2, f(t2,15,0.5), 'r', label='Gain = 15')
plt.plot(t2, f(t2,20,0.5), 'b', label='Gain = 20')

plt.legend()
plt.title('Sigmoid Correction: variation in gain')

plt.subplot(122)
plt.plot(t2, f(t2,15,0.4), 'g',label='Cutoff =0.4')
plt.plot(t2, f(t2,15,0.5), 'r',label='Cutoff =0.5')
plt.plot(t2, f(t2,15,0.6), 'b',label='Cutoff =0.6')

plt.title('Sigmoid Correction: variation in cutoff')
plt.legend()


plt.show()