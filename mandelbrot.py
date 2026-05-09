import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(h, w, max_iter=100):
    # Create a grid of complex numbers
    y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
    c = x + y*1j
    z = c
    # divtime keeps track of which iteration each point escaped
    divtime = max_iter + np.zeros(z.shape, dtype=int)

    for i in range(max_iter):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2            # Find points that have escaped
        div_now = diverge & (divtime == max_iter) # Find points escaping *now*
        divtime[div_now] = i                    # Note the iteration number
        z[diverge] = 2                          # Avoid large numbers for overflow

    return divtime

# Generate and plot
plt.figure(figsize=(10, 10))
fractal = mandelbrot(1000, 1000)
plt.imshow(fractal, cmap='magma', extent=[-2, 0.8, -1.4, 1.4])
plt.axis('off')

