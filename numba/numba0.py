import numpy as np
import numba
import time

def calculate_sum_python(arr):  # Pure Python version
    total = 0
    for i in range(arr.shape[0]):
        total += arr[i]
    return total

@numba.jit
def calculate_sum_numba(arr):   # Numba-compiled version
    total = 0
    for i in range(arr.shape[0]):
        total += arr[i]
    return total

# Small dataset
arr_small = np.arange(1000)

start_time = time.time()
result_python = calculate_sum_python(arr_small)
end_time = time.time()
print(f"Pure Python (small data): {end_time - start_time:.6f} seconds")

start_time = time.time()
result_numba = calculate_sum_numba(arr_small)
end_time = time.time()
print(f"Numba (small data): {end_time - start_time:.6f} seconds")

# Large dataset
arr_large = np.arange(10000000)  # 10 million elements

start_time = time.time()
result_python = calculate_sum_python(arr_large)
end_time = time.time()
print(f"Pure Python (large data): {end_time - start_time:.6f} seconds")

start_time = time.time()
result_numba = calculate_sum_numba(arr_large)
end_time = time.time()
print(f"Numba (large data): {end_time - start_time:.6f} seconds")

