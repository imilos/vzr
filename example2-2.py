from mpi4py import MPI
import numpy as np

def main():
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()

    N = 10000000

    # --- LOGIKA PO MICHAELU QUINNU ---
    # Vsak proces izračuna svoje meje v O(1) brez kumulativnih zank
    my_start = (my_rank * N) // world_size
    my_end   = ((my_rank + 1) * N) // world_size
    my_len   = my_end - my_start

    # 1. Inicializacija a
    start_time = MPI.Wtime()
    a = np.ones(my_len)
    end_time = MPI.Wtime()
    if my_rank == 0:
        print(f"Initialize a time: {end_time-start_time:.6f}s")

    # 2. Inicializacija b
    # Quinn logiko globalnih indeksov uporabimo za generiranje zaporedja
    start_time = MPI.Wtime()
    b = np.arange(my_start + 1, my_end + 1, dtype=float)
    end_time = MPI.Wtime()
    if my_rank == 0:
        print(f"Initialize b time: {end_time-start_time:.6f}s")

    # 3. Seštevanje nizov
    # Namesto 'for' zanke uporabimo NumPy vektorizacijo
    start_time = MPI.Wtime()
    a = a + b
    end_time = MPI.Wtime()
    if my_rank == 0:
        print(f"Add arrays time:    {end_time-start_time:.6f}s")

    # 4. Izračun povprečja
    start_time = MPI.Wtime()
    local_sum = np.array([np.sum(a)], dtype=float)
    world_sum = np.zeros(1, dtype=float)
    
    # Kolektivna komunikacija
    world_comm.Reduce(local_sum, world_sum, op=MPI.SUM, root=0)
    
    average = world_sum[0] / N
    end_time = MPI.Wtime()

    if my_rank == 0:
        print(f"Average result time: {end_time-start_time:.6f}s")
        print(f"Average: {average}")

if __name__ == "__main__":
    main()

