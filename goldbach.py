'''
Vsako sodo število, večje od 2, lahko predstavimo kot vsoto dveh praštevil.
'''
from mpi4py import MPI
import numpy as np

def block_low(id, p, n):
    """Izračuna spodnjo mejo bloka po Quinnovih pravilih."""
    return (id * (n + 1)) // p

def block_high(id, p, n):
    """Izračuna zgornjo mejo bloka po Quinnovih pravilih."""
    return block_low(id + 1, p, n) - 1

def get_sieve(n):
    """Eratostenovo sito za hitro iskanje praštevil."""
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0:2] = False
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            sieve[p*p : n+1 : p] = False
    return sieve

def check_goldbach_range(start, end, sieve):
    """Preveri Goldbachovo domnevo s preprostim skakanjem za 2."""
    # Začnemo pri prvem sodem številu v bloku, ki je vsaj 4
    first_even = start
    if first_even < 4:
        first_even = 4
    elif first_even % 2 != 0:
        first_even += 1

    for n in range(first_even, end + 1, 2):
        found = False
        for p1 in range(2, n // 2 + 1):
            if sieve[p1] and sieve[n - p1]:
                found = True
                break
        
        if not found:
            return False, n
    return True, None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    LIMIT = 10000000
    
    # --- Stari način z array_split (zakomentirano) ---
    # all_numbers = range(LIMIT + 1)
    # chunks = np.array_split(all_numbers, size)
    # my_chunk = chunks[rank]
    # local_start, local_end = my_chunk[0], my_chunk[-1]
    
    # --- Quinn standard na celotnem obsegu ---
    local_start = block_low(rank, size, LIMIT)
    local_end = block_high(rank, size, LIMIT)
    
    sieve = get_sieve(LIMIT)
    
    start_time = MPI.Wtime()
    success, failed_num = check_goldbach_range(local_start, local_end, sieve)
    results = comm.gather((success, failed_num), root=0)
    
    if rank == 0:
        end_time = MPI.Wtime()
        total_success = all(r[0] for r in results)
        
        print(f"--- Rezultati (Izvedeno na {size} procesih) ---")
        if total_success:
            print(f"Goldbachova domneva je POTRJENA do števila {LIMIT}.")
        else:
            failed_examples = [r[1] for r in results if r[1] is not None]
            print(f"Domneva NE drži. Prvi protiprimer: {min(failed_examples)}")
        
        print(f"Skupni čas izvajanja: {end_time - start_time:.4f} sekund")

if __name__ == "__main__":
    main()

