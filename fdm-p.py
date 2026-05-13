import numpy as np
import matplotlib.pyplot as plt
import math
from mpi4py import MPI

def block_low(id, p, n):
    return (id * n) // p

def block_high(id, p, n):
    return block_low(id + 1, p, n) - 1

def block_size(id, p, n):
    return block_high(id, p, n) - block_low(id, p, n) + 1

def heat_mpi(nt=10, nx=20, alpha=0.1, L=1, tmax=0.5):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parametri mreže
    dx = L / (nx - 1)
    dt = tmax / (nt - 1)
    r = alpha * dt / dx**2
    r2 = 1 - 2 * r

    # Dekompozicija po Quinnu
    my_nx = block_size(rank, size, nx)
    my_start = block_low(rank, size, nx)
    
    # Lokalna polja (vključno z robnimi točkami za izmenjavo - "ghost cells")
    # Vsak proces potrebuje prostor za my_nx + sosede
    u_local = np.zeros(my_nx + 2)
    u_next = np.zeros(my_nx + 2)

    # Začetni pogoji
    x_global = np.linspace(0, L, nx)
    for i in range(my_nx):
        idx_global = my_start + i
        u_local[i + 1] = math.sin(math.pi * x_global[idx_global] / L)

    # Glavna časovna zanka
    for m in range(1, nt):
        # Izmenjava robnih točk (Halo exchange)
        # Pošlji levo, prejmi od desne
        if rank > 0:
            comm.send(u_local[1], dest=rank - 1, tag=1)
            u_local[0] = comm.recv(source=rank - 1, tag=2)
        
        # Pošlji desno, prejmi od leve
        if rank < size - 1:
            comm.send(u_local[my_nx], dest=rank + 1, tag=2)
            u_local[my_nx + 1] = comm.recv(source=rank + 1, tag=1)

        # Izračun FTCS (le za notranje točke globalne domene)
        for i in range(1, my_nx + 1):
            curr_idx = my_start + i - 1
            if curr_idx == 0 or curr_idx == nx - 1:
                u_next[i] = 0 # Robni pogoji (U=0)
            else:
                u_next[i] = r * u_local[i - 1] + r2 * u_local[i] + r * u_local[i + 1]
        
        u_local[:] = u_next[:]

    # Zbiranje podatkov na procesu 0 za izpis in graf
    final_u_part = u_local[1:my_nx + 1]
    all_u = None
    if rank == 0:
        all_u = np.zeros(nx)
    
    # Uporabimo Gatherv, ker bloki morda niso enako veliki
    sendcounts = np.array([block_size(i, size, nx) for i in range(size)])
    displacements = np.array([block_low(i, size, nx) for i in range(size)])
    
    comm.Gatherv(sendbuf=final_u_part, recvbuf=[all_u, sendcounts, displacements, MPI.DOUBLE], root=0)

    if rank == 0:
        ue = np.sin(math.pi * x_global / L) * math.exp(-tmax * alpha * (math.pi / L)**2)
        err = np.linalg.norm(all_u - ue)
        print(f"Norm of error = {err:.6e} at t = {tmax}")
        print(f"dt = {dt:.6f}, dx = {dx:.6f}, r = {r:.6f}")

        plt.figure(figsize=(8, 5))
        plt.title('Vzporedni heatFTCS (MPI)')
        plt.plot(x_global, ue, 'r-', label='Exact')
        plt.plot(x_global, all_u, 'b--o', label='MPI FTCS')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    heat_mpi()
