import argparse
import numpy as np

# Nastavitve vrednosti za polje
ON = 255
OFF = 0
vals = [ON, OFF]

def randomGrid(N):
    """Vrne polje velikosti NxN z naključnimi vrednostmi."""
    return np.random.choice(vals, N*N, p=[0.2, 0.8]).reshape(N, N)

def addGlider(i, j, grid):
    """Doda 'glider' z zgornjim levim kotom na (i, j)."""
    glider = np.array([[0,    0, 255], 
                       [255,  0, 255], 
                       [0,  255, 255]])
    grid[i:i+3, j:j+3] = glider

def addGosperGliderGun(i, j, grid):
    """Doda Gosper Glider Gun z zgornjim levim kotom na (i, j)."""
    gun = np.zeros(11*38).reshape(11, 38)

    gun[5][1] = gun[5][2] = 255
    gun[6][1] = gun[6][2] = 255

    gun[3][13] = gun[3][14] = 255
    gun[4][12] = gun[4][16] = 255
    gun[5][11] = gun[5][17] = 255
    gun[6][11] = gun[6][15] = gun[6][17] = gun[6][18] = 255
    gun[7][11] = gun[7][17] = 255
    gun[8][12] = gun[8][16] = 255
    gun[9][13] = gun[9][14] = 255

    gun[1][25] = 255
    gun[2][23] = gun[2][25] = 255
    gun[3][21] = gun[3][22] = 255
    gun[4][21] = gun[4][22] = 255
    gun[5][21] = gun[5][22] = 255
    gun[6][23] = gun[6][25] = 255
    gun[7][25] = 255

    gun[3][35] = gun[3][36] = 255
    gun[4][35] = gun[4][36] = 255

    grid[i:i+11, j:j+38] = gun

def update(grid, N):
    """Izvede en korak simulacije (posodobitev stanja)."""
    newGrid = grid.copy()
    for i in range(N):
        for j in range(N):
            # Izračun vsote 8 sosedov s toroidnimi robnimi pogoji
            total = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] + 
                         grid[(i-1)%N, j] + grid[(i+1)%N, j] + 
                         grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] + 
                         grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N])/255)

            # Uporaba Conwayevih pravil
            if grid[i, j] == ON:
                if (total < 2) or (total > 3):
                    newGrid[i, j] = OFF
            else:
                if total == 3:
                    newGrid[i, j] = ON

    grid[:] = newGrid[:]
    return grid

def main():
    parser = argparse.ArgumentParser(description="Logika za Conway's Game of Life.")
    parser.add_argument('--grid-size', dest='N', type=int, default=100)
    parser.add_argument('--glider', action='store_true')
    parser.add_argument('--gosper', action='store_true')
    parser.add_argument('--steps', type=int, default=10, help="Stevilo korakov simulacije")
    
    args = parser.parse_args()
    N = args.N

    # Inicializacija polja
    if args.glider:
        grid = np.zeros(N*N).reshape(N, N)
        addGlider(1, 1, grid)
    elif args.gosper:
        grid = np.zeros(N*N).reshape(N, N)
        addGosperGliderGun(10, 10, grid)
    else:
        grid = randomGrid(N)

    # Primer poganjanja simulacije brez risanja
    for step in range(args.steps):
        grid = update(grid, N)
        # Tukaj bi lahko dodali npr. print(grid) ali shranjevanje v datoteko
        print(f"Korak {step+1} zakljucen.")

if __name__ == '__main__':
    main()

