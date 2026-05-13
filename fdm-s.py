import numpy as np
import matplotlib.pylab as plt
import math

def heatFTCS(nt=20, nx=20, alpha=0.1, L=1, tmax=0.5, errPlots=1):

    #heatFTCS Solve 1D heat equation with the FTCS scheme

    #Synopsis:  heatFTCS()
    #           heatFTCS(nt)
    #           heatFTCS(nt,nx)
    #           heatFTCS(nt,nx,alpha)
    #           heatFTCS(nt,nx,alpha,L)
    #           heatFTCS(nt,nx,alpha,L,tmax)
    #           heatFTCS(nt,nx,alpha,L,tmax,errPlots)
    #           err = heatFTCS(...)
    
    #Input:   nt = number of steps. Default: nt = 10;
    #         nx = number of mesh points in x direction. Default: nx=20
    #         alpha = diffusivity. Default: alpha = 0.1
    #         L = length of the domain. Default: L = 1;
    #         tmax = maximum time for the simulation. Default: tmax = 0.5
    #         errPlots = flag (1 or 0) to control whether error plots should be shown
    
    #Output: err = L2 norm of error evaluated at the spatial nodes on last time step
    #        x = location of finite difference nodes
    #        t = values of time at which solution is obtained (time nodes)
    #        U = matrix of solutions: U(:,j) is U(x) at t = t(j)

    #Compute mesh spacing and time step
    dx = L/(nx-1)
    dt = tmax/(nt-1)
    r = alpha*dt / dx**2
    r2 = 1 - 2*r
        
    #Create arrays to save data for export
    x = np.linspace(0, L, nx)
    t = np.linspace(0, tmax, nt)
    u = np.zeros((nx,nt), dtype=float)

    #Set IC and BC
    q = np.zeros(nx)
    z = math.pi * x / L

    for i in range(0,nx):
        q[i] = math.sin(z[i])

    u[:,0] = q
    u0 = 0
    uL = 0

    #Loop over time steps
    for m in range(1, nt):
        for i in range(1, nx-1):
            u[i,m] = r*u[i-1, m-1] + r2*u[i, m-1] + r*u[i+1, m-1]

    #Compare with exact solution at end of simulation
    ue = q * math.exp(-t[nt-1] * alpha * (math.pi / L)**2)
    err = np.linalg.norm(u[:, nt-1] - ue)
    errgraph = u[:, nt-1] - ue

    print('Norm of error = ', err, ' at t = ', t[nt-1])
    print('tdt = ', dt, ' dx = ', dx, ' r = ', r)

    plt.title('heatFTCS', fontsize=20)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('u', fontsize=15)
    plt.plot(x, ue, 'r-', linewidth=1, label='Exact')
    plt.plot(x, u[:,nt-1], 'b--o', linewidth=1, label='FTCS')
    plt.legend()
    return plt.show()


