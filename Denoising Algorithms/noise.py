import numpy as np

def noise(inp, mean=0, stddev=0.1):
    
    noisy = inp + stddev * np.random.standard_normal(inp.shape)
    noisy = np.clip(noisy, 0, 1)
    
    return noisy