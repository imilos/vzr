import numpy as np
from mpi4py import MPI

# Quinn-ove makro funkcije za dekompozicijo
def BLOCK_LOW(id, p, n):
    return (id * n) // p

def BLOCK_HIGH(id, p, n):
    return BLOCK_LOW(id + 1, p, n) - 1

def BLOCK_SIZE(id, p, n):
    return BLOCK_HIGH(id, p, n) - BLOCK_LOW(id, p, n) + 1

def update_local(local_grid, N, local_h):
    """Posodobi lokalni del polja (vključno s halo vrsticami)."""
    # local_grid ima dimenzije (local_h + 2, N)
    new_grid = local_grid.copy()
    for i in range(1, local_h + 1):
        for j in range(N):
            # Vsota 8 sosedov (vrstice so lokalne, stolpci so toroidni)
            total = int((local_grid[i, (j-1)%N] + local_grid[i, (j+1)%N] +
                         local_grid[i-1, j] + local_grid[i+1, j] +
                         local_grid[i-1, (j-1)%N] + local_grid[i-1, (j+1)%N] +
                         local_grid[i+1, (j-1)%N] + local_grid[i+1, (j+1)%N]) / 255)

            if local_grid[i, j] == 255:
                if (total < 2) or (total > 3):
                    new_grid[i, j] = 0
            else:
                if total == 3:
                    new_grid[i, j] = 255
    return new_grid

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 1000  # Velikost celotnega polja
    steps = 10
    
    # Izračun lokalnih meja
    local_h = BLOCK_SIZE(rank, size, N)
    
    # Vsak proces ima svoje vrstice + 2 vrstici za halo (zgoraj in spodaj)
    if rank == 0:
        # Samo rank 0 inicializira celotno polje (za preprostost primera)
        full_grid = np.random.choice([0, 255], N*N, p=[0.8, 0.2]).reshape(N, N)
    else:
        full_grid = None

    # Razpošiljanje (Scatter) podatkov procesom
    # Ker so bloki lahko različnih velikosti, bi v praksi uporabili Scatterv, 
    # tukaj pa za preprostost predpostavimo inicializacijo na vsakem procesu posebej:
    local_data = np.random.choice([0, 255], local_h * N, p=[0.8, 0.2]).reshape(local_h, N)
    
    # Ustvarimo lokalno polje s prostorom za halo vrstice
    grid = np.zeros((local_h + 2, N), dtype=int)
    grid[1:local_h+1, :] = local_data

    for step in range(steps):
        # Zamenjava mejnih vrstic (Halo Exchange)
        top_neighbor = (rank - 1) % size
        bottom_neighbor = (rank + 1) % size

        # Pošljemo spodnjo vrstico sosedu dol, sprejmemo v zgornjo halo vrstico
        comm.Sendrecv(grid[local_h, :], dest=bottom_neighbor, 
                      recvbuf=grid[0, :], source=top_neighbor)
        
        # Pošljemo zgornjo vrstico sosedu gor, sprejmemo v spodnjo halo vrstico
        comm.Sendrecv(grid[1, :], dest=top_neighbor, 
                      recvbuf=grid[local_h+1, :], source=bottom_neighbor)

        # Izračun novega stanja
        grid = update_local(grid, N, local_h)
        
        if rank == 0:
            print(f"Korak {step+1} zakljucen.")

    comm.Barrier()

if __name__ == '__main__':
    main()

