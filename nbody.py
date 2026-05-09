import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
N_TOTAL = 1000
N_LOCAL = N_TOTAL // size
DT = 0.01
G = 1.0

# Initialize particles on each rank (ideally from a seed for consistency)
np.random.seed(42 + rank)
local_pos = np.random.rand(N_LOCAL, 3).astype('f')
local_vel = np.zeros((N_LOCAL, 3), dtype='f')
local_mass = np.ones(N_LOCAL, dtype='f')

def compute_forces(local_pos, all_pos, all_mass):
    forces = np.zeros_like(local_pos)
    for i in range(len(local_pos)):
        # Vectorized distance calculation
        diff = all_pos - local_pos[i]
        dist = np.sqrt(np.sum(diff**2, axis=1)) + 1e-9 # softening factor
        dist_inv3 = (dist**(-3))
        forces[i] = G * np.sum(diff * (all_mass * dist_inv3)[:, np.newaxis], axis=0)
    return forces

# Simulation Loop
for step in range(100):
    # 1. Gather all positions to every rank
    all_pos = np.empty((N_TOTAL, 3), dtype='f')
    comm.Allgather(local_pos, all_pos)
    #print(local_pos.shape, all_pos.shape)
    
    # In a real scenario, you'd also gather masses if they differ
    all_mass = np.ones(N_TOTAL, dtype='f') 

    # 2. Compute forces for local particles
    forces = compute_forces(local_pos, all_pos, all_mass)

    # 3. Update velocity and position (Euler integration)
    local_vel += forces * DT
    local_pos += local_vel * DT

if rank == 0:
    print("Simulation complete.")

