'''
Koda v Pythonu serijsko (zaporedno) obdela 10 milijonov števil. Najprej pripravi dva vektorja, ju element po elementu sešteje, 
nato pa v zanki izračuna povprečje vseh dobljenih vrednosti. Ker so zanke for v Pythonu pri takšni količini podatkov počasne, 
koda služi kot idealna osnova za vajo iz paralelizacije: študenti jo morajo razbiti na dele, da vsak proces izračuna le svoj 
kos vsote, nato pa rezultate združijo z MPI komunikacijo.
'''

from mpi4py import MPI
import numpy as np

if __name__ == "__main__":

    N = 10000000

    # initialize a
    a = np.ones( N )

    # initialize b
    b = np.zeros( N )
    for i in range( N ):
        b[i] = 1.0 + i

    # add the two arrays
    for i in range( N ):
        a[i] = a[i] + b[i]

    # average the result
    sum = 0.0
    for i in range( N ):
        sum += a[i]
    average = sum / N

    print("Average: " + str(average))
