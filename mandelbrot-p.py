import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

# --- Logika za dekompozicijo (Prevedeni C makri) ---
def BLOCK_LOW(id, p, n):
    return (id * n) // p

def BLOCK_HIGH(id, p, n):
    return BLOCK_LOW(id + 1, p, n) - 1

def BLOCK_SIZE(id, p, n):
    return BLOCK_HIGH(id, p, n) - BLOCK_LOW(id, p, n) + 1

# --- Nastavitve ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

W, H = 1000, 1000
max_iter = 100

# Določanje obsega vrstic za trenutni proces
my_start = BLOCK_LOW(rank, size, H)
my_end = BLOCK_HIGH(rank, size, H)
my_height = BLOCK_SIZE(rank, size, H)

# --- Računanje (Lokalni blok) ---
# Ustvarimo mrežo samo za vrstice, ki pripadajo temu procesu
y, x = np.ogrid[-1.4:1.4:H*1j, -2:0.8:W*1j]
y_local = y[my_start:my_end+1] # Izločimo lokalne vrstice
c = x + y_local*1j
z = c
divtime = max_iter + np.zeros(z.shape, dtype=int)

for i in range(max_iter):
    z = z**2 + c
    diverge = z.real**2 + z.imag**2 > 4
    div_now = diverge & (divtime == max_iter)
    divtime[div_now] = i
    z[diverge] = 2

# --- Zbiranje podatkov (Gather) ---
# rank 0 zbere vse dele v seznam, ostali pošljejo svoje
all_parts = comm.gather(divtime, root=0)

# --- Prikaz (Samo na rank 0) ---
#if rank == 0:
    # Združevanje delov v eno veliko sliko
#    full_fractal = np.vstack(all_parts)
    
#    plt.figure(figsize=(10, 10))
#    plt.imshow(full_fractal, cmap='magma', extent=[-2, 0.8, -1.4, 1.4])
#    plt.axis('off')
#    plt.show()

