# exercise_1/parallel_row.py
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

def multiply_rows(args):
    """Cada worker multiplica su bloque de filas por B."""
    A_chunk, B = args
    return A_chunk @ B

def parallel_matmul_rows(A, B, num_workers):
    row_chunks = np.array_split(A, num_workers, axis=0)
    with Pool(processes=num_workers) as pool:
        results = pool.map(multiply_rows, [(chunk, B) for chunk in row_chunks])
    return np.vstack(results)

def run_benchmark(sizes=[128, 256, 512, 1024], workers_list=[2, 4]):
    print(f"{'Tamaño':>8} {'Workers':>8} {'T paralelo':>12} {'T serial':>10} {'Speedup':>9} {'Eficiencia':>11}")
    print("-" * 65)
    for n in sizes:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)

        # Serial reference
        t0 = time.perf_counter()
        _ = A @ B
        t_serial = time.perf_counter() - t0

        for w in workers_list:
            t0 = time.perf_counter()
            C = parallel_matmul_rows(A, B, w)
            t_par = time.perf_counter() - t0

            speedup = t_serial / t_par if t_par > 0 else 0
            efficiency = speedup / w
            print(f"{n:>8} {w:>8} {t_par:>12.4f} {t_serial:>10.4f} {speedup:>9.2f} {efficiency:>11.2f}")

if __name__ == "__main__":
    # Validación
    A = np.random.rand(64, 64)
    B = np.random.rand(64, 64)
    C_serial = A @ B
    C_parallel = parallel_matmul_rows(A, B, num_workers=2)
    assert np.allclose(C_serial, C_parallel), "❌ Error en resultado paralelo"
    print("✅ Validación correcta\n")

    run_benchmark()