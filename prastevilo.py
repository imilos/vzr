"""
Preverjanje praštevil z naivno metodo (trial division) - vzporedna implementacija z mpi4py
Uporaba: mpiexec -n <p> python prime_check_mpi.py <n>
"""

import sys
import math
from mpi4py import MPI

def is_prime(num):
    """Preveri, če je število praštevilo z naivnim preverjanjem deljivosti."""
    if num < 2:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    
    # Preverjamo le liha števila do korena števila n
    limit = int(math.sqrt(num))
    for i in range(3, limit + 1, 2):
        if num % i == 0:
            return False
    return True

def block_low(id, p, n):
    return (id * n) // p

def block_high(id, p, n):
    return block_low(id + 1, p, n) - 1

def main():
    comm = MPI.COMM_WORLD
    id   = comm.Get_rank()
    p    = comm.Get_size()

    # Merjenje časa
    comm.Barrier()
    start_time = MPI.Wtime()

    if len(sys.argv) != 2:
        if id == 0:
            print(f"Uporaba: mpiexec -n <p> python {sys.argv[0]} <n>")
        MPI.Finalize()
        sys.exit(1)

    n = int(sys.argv[1])

    # Razdelimo območje [2, n] med procese
    # Celotno število elementov je (n - 1), ker začnemo pri 2
    total_elements = n - 1
    
    low_val  = 2 + block_low(id, p, total_elements)
    high_val = 2 + block_high(id, p, total_elements)

    # Vsak proces prešteje praštevila v svojem dodeljenem bloku
    local_count = 0
    #for i in range(low_val, high_val + 1):
    for i in range(id, n, p):
        if is_prime(i):
            local_count += 1

    # Zberemo rezultate vseh procesov na procesu 0
    global_count = comm.reduce(local_count, op=MPI.SUM, root=0)
    
    end_time = MPI.Wtime()
    elapsed_time = end_time - start_time

    if id == 0:
        print(f"Najdenih prastevil do {n}: {global_count}")
        print(f"Cas izvajanja: {elapsed_time:.6f} sekund")

    MPI.Finalize()

if __name__ == "__main__":
    main()

