import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import time
import os

# Estados: 0=no-burnable, 1=susceptible, 2=burning, 3=burned
CMAP = mcolors.ListedColormap(["#888888", "#2d8a2d", "#ff2200", "#1a1a1a"])

def step(grid, frp_grid, burn_prob=0.3, lifetime=3, burn_counter=None):
    if burn_counter is None:
        burn_counter = np.zeros_like(grid)
    new_grid = grid.copy()
    new_counter = burn_counter.copy()
    rows, cols = grid.shape

    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if grid[r, c] == 1:
                neighbors = grid[r-1:r+2, c-1:c+2]
                if np.sum(neighbors == 2) > 0:
                    frp_factor = min(frp_grid[r, c] / 10.0, 1.0)
                    p = burn_prob * np.sum(neighbors == 2) + 0.1 * frp_factor
                    if np.random.random() < p:
                        new_grid[r, c] = 2
                        new_counter[r, c] = 0
            elif grid[r, c] == 2:
                new_counter[r, c] += 1
                if burn_counter[r, c] >= lifetime:
                    new_grid[r, c] = 3
    return new_grid, new_counter

def run_serial_sim(n_steps=20):
    # Cargar datos
    df = pd.read_csv("exercise_3/firms_data.csv")
    GRID_ROWS, GRID_COLS = 200, 200
    lat_min, lat_max = 17.5, 21.5
    lon_min, lon_max = -91.5, -86.5

    def latlon_to_grid(lat, lon):
        r = int((lat - lat_min) / (lat_max - lat_min) * GRID_ROWS)
        c = int((lon - lon_min) / (lon_max - lon_min) * GRID_COLS)
        return np.clip(r, 0, GRID_ROWS-1), np.clip(c, 0, GRID_COLS-1)

    grid = np.ones((GRID_ROWS, GRID_COLS), dtype=np.int8)
    frp_grid = np.zeros((GRID_ROWS, GRID_COLS))
    for _, row in df.iterrows():
        ri, ci = latlon_to_grid(row["latitude"], row["longitude"])
        grid[ri, ci] = 2
        frp_grid[ri, ci] = row["frp"]

    counter = np.zeros_like(grid)
    t0 = time.perf_counter()
    for i in range(n_steps):
        grid, counter = step(grid, frp_grid, burn_counter=counter)
    t_elapsed = time.perf_counter() - t0
    print(f"Simulación finalizada en {t_elapsed:.2f}s")
    
    # Guardar estado final
    plt.imshow(grid, cmap=CMAP, origin="lower")
    plt.title(f"Estado final después de {n_steps} pasos")
    os.makedirs("exercise_3/frames", exist_ok=True)
    plt.savefig("exercise_3/frames/final_state.png")
    print("Estado final guardado en exercise_3/frames/final_state.png")

if __name__ == "__main__":
    run_serial_sim()
