# exercise_1/serial_matmul.py
import os
# Desactivar el multithreading interno de BLAS/NumPy para una comparación justa
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import time

def serial_matmul(A, B):
    """Multiplicación serial pura con numpy (baseline)."""
    return A @ B

def run_benchmark(sizes=[128, 256, 512, 1024]):
    print(f"{'Tamaño':>10} {'Tiempo (s)':>12}")
    print("-" * 25)
    for n in sizes:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)

        start = time.perf_counter()
        C = serial_matmul(A, B)
        elapsed = time.perf_counter() - start

        print(f"{n:>10} {elapsed:>12.4f}")

if __name__ == "__main__":
    # Validación de correctitud
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = serial_matmul(A, B)
    expected = np.array([[19, 22], [43, 50]])
    assert np.allclose(C, expected), "❌ Error en multiplicación"
    print("✅ Validación correcta\n")

    run_benchmark()