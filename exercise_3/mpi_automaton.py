import numpy as np
from mpi4py import MPI
import time
import pandas as pd
import os

def latlon_to_grid(lat, lon, lat_min, lat_max, lon_min, lon_max, grid_rows, grid_cols):
    r = int((lat - lat_min) / (lat_max - lat_min) * grid_rows)
    c = int((lon - lon_min) / (lon_max - lon_min) * grid_cols)
    return np.clip(r, 0, grid_rows-1), np.clip(c, 0, grid_cols-1)

def run_mpi_automaton():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    GRID_ROWS, GRID_COLS = 200, 200
    N_STEPS = 20
    BURN_PROB = 0.3
    LIFETIME = 3
    lat_min, lat_max = 17.5, 21.5
    lon_min, lon_max = -91.5, -86.5
    
    # Process 0 constructs the initial grid
    if rank == 0:
        if os.path.exists("exercise_3/firms_data.csv"):
            df = pd.read_csv("exercise_3/firms_data.csv")
        elif os.path.exists("firms_data.csv"):
            df = pd.read_csv("firms_data.csv")
        else:
            print("Datos FIRMS no encontrados. Abortando.")
            df = None
            comm.Abort(1)
            
        grid_full = np.ones((GRID_ROWS, GRID_COLS), dtype=np.int8)
        frp_full = np.zeros((GRID_ROWS, GRID_COLS))
        
        for _, row in df.iterrows():
            ri, ci = latlon_to_grid(row["latitude"], row["longitude"], lat_min, lat_max, lon_min, lon_max, GRID_ROWS, GRID_COLS)
            grid_full[ri, ci] = 2
            frp_full[ri, ci] = row["frp"]
    else:
        grid_full = None
        frp_full = None
        
    # Broadcast full structures (simplified implementation, normally scatter rows)
    grid_full = comm.bcast(grid_full, root=0)
    frp_full = comm.bcast(frp_full, root=0)
    
    rows_per_proc = GRID_ROWS // size
    start_row = rank * rows_per_proc
    end_row = (rank + 1) * rows_per_proc if rank != size - 1 else GRID_ROWS
    local_rows = end_row - start_row
    
    # Local grids with halos
    local_grid = np.ones((local_rows + 2, GRID_COLS), dtype=np.int8)
    local_frp = np.zeros((local_rows + 2, GRID_COLS))
    local_counter = np.zeros((local_rows + 2, GRID_COLS), dtype=np.int8)
    
    # Copy assigned rows
    local_grid[1:-1, :] = grid_full[start_row:end_row, :]
    local_frp[1:-1, :] = frp_full[start_row:end_row, :]
    
    # Copy counters for already burning
    local_counter[local_grid == 2] = 0
    
    np.random.seed(42 + rank)
    
    comm.Barrier()
    t0 = MPI.Wtime()
    
    for step in range(N_STEPS):
        # Exchange halos
        reqs = []
        if rank > 0:
            reqs.append(comm.Isend(local_grid[1, :].copy(), dest=rank-1, tag=11))
            reqs.append(comm.Irecv(local_grid[0, :], source=rank-1, tag=22))
        if rank < size - 1:
            reqs.append(comm.Isend(local_grid[-2, :].copy(), dest=rank+1, tag=22))
            reqs.append(comm.Irecv(local_grid[-1, :], source=rank+1, tag=11))
            
        MPI.Request.Waitall(reqs)
        
        # Local Step
        new_grid = local_grid.copy()
        new_counter = local_counter.copy()
        
        for r in range(1, local_rows + 1):
            for c in range(1, GRID_COLS - 1):
                if local_grid[r, c] == 1:
                    # Susceptible
                    neighbors = local_grid[r-1:r+2, c-1:c+2]
                    burning_neighbors = np.sum(neighbors == 2)
                    if burning_neighbors > 0:
                        frp_factor = min(local_frp[r, c] / 10.0, 1.0)
                        p = BURN_PROB * burning_neighbors + 0.1 * frp_factor
                        if np.random.random() < p:
                            new_grid[r, c] = 2
                            new_counter[r, c] = 0
                elif local_grid[r, c] == 2:
                    # Burning
                    new_counter[r, c] += 1
                    if local_counter[r, c] >= LIFETIME:
                        new_grid[r, c] = 3
                        
        local_grid = new_grid
        local_counter = new_counter
        
    comm.Barrier()
    t_par = MPI.Wtime() - t0
    
    # Gather results (without halos)
    gathered_grids = comm.gather(local_grid[1:-1, :], root=0)
    
    if rank == 0:
        final_grid = np.vstack(gathered_grids)
        burning = np.sum(final_grid == 2)
        burned = np.sum(final_grid == 3)
        susceptible = np.sum(final_grid == 1)
        print(f"--- Ejercicio 3: Autómata Forestal MPI ---")
        print(f"Procesos: {size}, Pasos: {N_STEPS}")
        print(f"Tiempo: {t_par:.4f}s")
        print(f"Estado final -> 🔥 Ardiendo: {burning}, ⬛ Quemado: {burned}, 🌿 Susceptible: {susceptible}")

if __name__ == "__main__":
    run_mpi_automaton()