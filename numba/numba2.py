import numpy as np
import math
from numba import njit, prange

# 1. Uporabimo njit (skrajšano za nopython=True)
# 2. parallel=True vklopi samodejno paralelizacijo
#@njit(parallel=True)
def func_parallel(a, b):
    size = a.shape[0]
    result = np.empty(size, dtype=np.float64)
    
    # Namesto range uporabimo prange, ki delo 
    # samodejno razdeli med vsa razpoložljiva jedra
    #for i in prange(size):
    for i in range(size):
        result[i] = math.exp(2.1 * math.cos(a[i]) + 3.2 * math.sin(b[i]))
        
    return result

# Uporaba je zdaj popolnoma enostavna
size = 10**8
a = np.random.rand(size)
b = np.random.rand(size)

# To bo samodejno uporabilo vsa jedra (npr. 4)
result = func_parallel(a, b)

