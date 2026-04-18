from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Preverimo, ali imamo natanko 2 procesa
if size != 2:
    if rank == 0:
        print("Ta primer zahteva natanko 2 procesa.")
    exit()

ping_pong_count = 0
partner_rank = (rank + 1) % 2
limit = 10  # Število udarcev

while ping_pong_count < limit:
    if rank == ping_pong_count % 2:
        # Povečamo števec pred pošiljanjem
        ping_pong_count += 1
        comm.send(ping_pong_count, dest=partner_rank)
        print(f"Proces {rank} poslal ping-pong {ping_pong_count} procesu {partner_rank}")
    else:
        # Čakamo na sprejem
        ping_pong_count = comm.recv(source=partner_rank)
        print(f"Proces {rank} prejel ping-pong {ping_pong_count} od procesa {partner_rank}")

# Zagon: mpirun -np 2 python ping_pong.py

