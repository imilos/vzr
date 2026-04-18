from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

token = -1

if rank != 0:
    # Vsi procesi razen 0 najprej čakajo na žeton od predhodnika (rank - 1)
    token = comm.recv(source=rank - 1, tag=0)
    print(f"Proces {rank} je prejel zeton {token} od procesa {rank - 1}")
else:
    # Proces 0 nastavi začetno vrednost žetona
    token = 100
    print(f"Proces 0 zacenja krog z zetonom {token}")

# Vsak proces pošlje žeton naslednjemu (rank + 1). 
# Zadnji proces ga pošlje nazaj na 0 zaradi operacije modulo.
dest = (rank + 1) % size
comm.send(token, dest=dest, tag=0)

# Proces 0 na koncu počaka, da žeton zaključi cel krog
if rank == 0:
    token = comm.recv(source=size - 1, tag=0)
    print(f"Proces 0 je prejel zeton {token} nazaj od zadnjega procesa {size - 1}")

