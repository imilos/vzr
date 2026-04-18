from mpi4py import MPI
import numpy as np

if __name__ == "__main__":

    # get basic information about the MPI communicator
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()

    N = 10000000

    # --- LOGIKA PO MICHAELU QUINNU (BLOCK_LOW in BLOCK_HIGH) ---
    # Vsak proces izračuna svoj začetek in dolžino neposredno
    my_start = (my_rank * N) // world_size
    my_end   = ((my_rank + 1) * N) // world_size
    my_len   = my_end - my_start

    # initialize a
    start_time = MPI.Wtime()
    a = np.ones(my_len)
    end_time = MPI.Wtime()
    if my_rank == 0:
        print("Initialize a time: " + str(end_time-start_time))

    # initialize b
    start_time = MPI.Wtime()
    b = np.zeros(my_len)
    for i in range(my_len):
        b[i] = 1.0 + (i + my_start)
    end_time = MPI.Wtime()
    if my_rank == 0:
        print("Initialize b time: " + str(end_time-start_time))

    # add the two arrays
    start_time = MPI.Wtime()
    for i in range(my_len):
        a[i] = a[i] + b[i]
    end_time = MPI.Wtime()
    if my_rank == 0:
        print("Add arrays time: " + str(end_time-start_time))

    # average the result
    start_time = MPI.Wtime()
    sum_val = 0.0
    for i in range(my_len):
        sum_val += a[i]
    
    # Priprava za MPI Reduce
    send_buf = np.array([sum_val], dtype=float)
    recv_buf = np.zeros(1, dtype=float)
    
    world_comm.Reduce([send_buf, MPI.DOUBLE], [recv_buf, MPI.DOUBLE], op=MPI.SUM, root=0)
    
    average = recv_buf[0] / N
    end_time = MPI.Wtime()
    
    if my_rank == 0:
        print("Average result time: " + str(end_time-start_time))
        print("Average: " + str(average))

