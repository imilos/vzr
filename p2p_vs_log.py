'''
Študenti morajo implementirati program v Pythonu z uporabo knjižnice mpi4py, ki primerja časovno učinkovitost seštevanja 
velikega polja števil (npr. 10^6 elementov) na dva načina: najprej z ročno implementacijo, kjer korensko vozlišče v zanki 
zaporedno prejema podatke od vseh ostalih procesov s comm.Send in comm.Recv, ter nato z uporabo vgrajene kolektivne 
funkcije comm.Reduce. Program mora izmeriti trajanje obeh operacij, študenti pa morajo meritve ponoviti pri različnem 
številu procesov (npr. 2, 4, 8, 12, 16, 32, 64, 128), podatke prenesti v LibreOffice Calc ali Excel, tam izrisati graf 
odvisnosti časa od števila procesov ter v kratkem poročilu na podlagi rezultatov razložiti, zakaj je Reduce pri večjem 
številu procesov hitrejši od linearnega pristopa P2P.
'''
from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Priprava podatkov (npr. niz 10 milijona števil)
n = 10000000
data = np.full(n, rank, dtype='f')
result_p2p = np.zeros(n, dtype='f')
result_reduce = np.zeros(n, dtype='f')

# --- 1. Pristop: Ročno pošiljanje (P2P) ---
comm.Barrier()  # Sinhronizacija pred merjenjem
start_p2p = time.time()

if rank == 0:
    result_p2p = data.copy()
    for i in range(1, size):
        temp = np.empty(n, dtype='f')
        comm.Recv([temp, MPI.FLOAT], source=i, tag=11)
        result_p2p += temp
else:
    comm.Send([data, MPI.FLOAT], dest=0, tag=11)

end_p2p = time.time()

# --- 2. Pristop: Kolektivna operacija (Reduce) ---
comm.Barrier()
start_reduce = time.time()

# comm.Reduce je optimiziran in pogosto uporablja drevesni algoritem
comm.Reduce([data, MPI.FLOAT], [result_reduce, MPI.FLOAT], op=MPI.SUM, root=0)

end_reduce = time.time()

# Izpis rezultatov na korenskem procesu
if rank == 0:
    print(f"Stevilo procesov: {size}")
    print(f"P2P cas:    {end_p2p - start_p2p:.6f} s")
    print(f"Reduce cas: {end_reduce - start_reduce:.6f} s")
    
    # Preverjanje pravilnosti
    if np.allclose(result_p2p, result_reduce):
        print("Oba rezultata sta identicna.")

