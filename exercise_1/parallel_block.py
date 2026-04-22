# exercise_1/parallel_block.py
import os
# Desactivar el multithreading interno de BLAS/NumPy para una comparación justa
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import time
from multiprocessing import Pool

def multiply_block(args):
    """Cada worker calcula un bloque C[i,j] = suma de A[i,k] @ B[k,j]."""
    row_start, row_end, col_start, col_end, A, B = args
    return (row_start, col_start, A[row_start:row_end, :] @ B[:, col_start:col_end])

def parallel_matmul_blocks(A, B, num_workers=4):
    n = A.shape[0]
    # Dividir en grilla de bloques (intentamos grilla cuadrada)
    grid = max(1, int(np.sqrt(num_workers)))
    row_splits = np.array_split(range(n), grid)
    col_splits = np.array_split(range(n), grid)

    tasks = []
    for rs in row_splits:
        for cs in col_splits:
            tasks.append((rs[0], rs[-1]+1, cs[0], cs[-1]+1, A, B))

    with Pool(processes=num_workers) as pool:
        results = pool.map(multiply_block, tasks)

    C = np.zeros((n, n))
    for (r0, c0, block) in results:
        r1 = r0 + block.shape[0]
        c1 = c0 + block.shape[1]
        C[r0:r1, c0:c1] = block
    return C

def run_benchmark(sizes=[128, 256, 512, 1024], workers_list=[2, 4]):
    print(f"{'Tamaño':>8} {'Workers':>8} {'T paralelo':>12} {'T serial':>10} {'Speedup':>9} {'Eficiencia':>11}")
    print("-" * 65)
    for n in sizes:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)

        t0 = time.perf_counter()
        _ = A @ B
        t_serial = time.perf_counter() - t0

        for w in workers_list:
            t0 = time.perf_counter()
            C = parallel_matmul_blocks(A, B, w)
            t_par = time.perf_counter() - t0

            speedup = t_serial / t_par if t_par > 0 else 0
            efficiency = speedup / w
            print(f"{n:>8} {w:>8} {t_par:>12.4f} {t_serial:>10.4f} {speedup:>9.2f} {efficiency:>11.2f}")

if __name__ == "__main__":
    A = np.random.rand(64, 64)
    B = np.random.rand(64, 64)
    C_serial = A @ B
    C_parallel = parallel_matmul_blocks(A, B, num_workers=4)
    assert np.allclose(C_serial, C_parallel), "❌ Error en resultado paralelo"
    print("✅ Validación correcta\n")

    run_benchmark()