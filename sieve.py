"""
Sieve of Eratosthenes parallel implementation using mpi4py
Usage: mpiexec -n <p> python sieve_mpi.py <n>
"""

import sys
import math
from mpi4py import MPI


def block_low(id, p, n):
    return (id * n) // p

def block_high(id, p, n):
    return block_low(id + 1, p, n) - 1

def block_size(id, p, n):
    return block_high(id, p, n) - block_low(id, p, n) + 1


def main():
    comm = MPI.COMM_WORLD
    id   = comm.Get_rank()
    p    = comm.Get_size()

    comm.Barrier()
    elapsed_time = -MPI.Wtime()

    if len(sys.argv) != 2:
        if id == 0:
            print(f"Command line: {sys.argv[0]} <n>")
        MPI.Finalize()
        sys.exit(1)

    n = int(sys.argv[1])

    # Break [2, n] into p blocks
    low_value  = 2 + block_low (id, p, n - 1)
    high_value = 2 + block_high(id, p, n - 1)
    size       = block_size    (id, p, n - 1)

    proc0_size = (n - 1) // p

    # All prime divisors of numbers < n must reside in process 0's block
    if (1 + proc0_size) < int(math.sqrt(n)):
        if id == 0:
            print("Too many processes")
        MPI.Finalize()
        sys.exit(1)

    # Each element represents whether the corresponding number is marked (composite)
    marked = bytearray(size)   # all zeros → all potentially prime

    index = 0
    prime = 2

    while True:
        # Find the first multiple of `prime` that falls in this process' range
        if prime * prime > low_value:
            first = prime * prime - low_value
        else:
            if low_value % prime == 0:
                first = 0
            else:
                first = prime - (low_value % prime)

        # Mark composites
        for i in range(first, size, prime):
            marked[i] = 1

        # Process 0 finds the next prime and broadcasts it
        if id == 0:
            index += 1
            while marked[index]:
                index += 1
            prime = index + 2          # index 0 → value 2, so value = index + 2

        prime = comm.bcast(prime, root=0)

        if prime * prime >= n:
            break

    # Count local unmarked (prime) numbers
    count = marked.count(0)

    # Sum across all processes
    global_count = comm.reduce(count, op=MPI.SUM, root=0)

    elapsed_time += MPI.Wtime()

    if id == 0:
        print(f"{global_count} primes are less than or equal to {n}")
        print(f"Total elapsed time: {elapsed_time:10.6f}")

    MPI.Finalize()


if __name__ == "__main__":
    main()


