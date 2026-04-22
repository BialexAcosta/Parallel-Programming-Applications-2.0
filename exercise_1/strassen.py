import numpy as np
import time

def strassen(A, B):
    n = A.shape[0]
    # Caso base
    if n <= 64:
        return A @ B

    mid = n // 2
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]

    # 7 productos en lugar de 8
    M1 = strassen(A11 + A22, B11 + B22)
    M2 = strassen(A21 + A22, B11)
    M3 = strassen(A11,        B12 - B22)
    M4 = strassen(A22,        B21 - B11)
    M5 = strassen(A11 + A12, B22)
    M6 = strassen(A21 - A11, B11 + B12)
    M7 = strassen(A12 - A22, B21 + B22)

    C = np.zeros((n, n))
    C[:mid, :mid] = M1 + M4 - M5 + M7
    C[:mid, mid:] = M3 + M5
    C[mid:, :mid] = M2 + M4
    C[mid:, mid:] = M1 - M2 + M3 + M6
    return C

def pad_to_power_of_2(A):
    n = max(A.shape)
    p = 1
    while p < n:
        p *= 2
    padded = np.zeros((p, p))
    padded[:A.shape[0], :A.shape[1]] = A
    return padded

def run_benchmark(sizes=[128, 256, 512]):  # 1024 es muy lento en Strassen puro
    print(f"{'Tamaño':>8} {'T serial':>10} {'T strassen':>12} {'Speedup':>9}")
    print("-" * 45)
    for n in sizes:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        Ap = pad_to_power_of_2(A)
        Bp = pad_to_power_of_2(B)

        t0 = time.perf_counter()
        C_serial = A @ B
        t_serial = time.perf_counter() - t0

        t0 = time.perf_counter()
        C_strassen = strassen(Ap, Bp)[:n, :n]
        t_strassen = time.perf_counter() - t0

        assert np.allclose(C_serial, C_strassen, atol=1e-6), \"❌ Error\"
        speedup = t_serial / t_strassen
        print(f\"{n:>8} {t_serial:>10.4f} {t_strassen:>12.4f} {speedup:>9.2f}\")

if __name__ == \"__main__\":
    # Validación
    A = np.array([[1,2],[3,4]], dtype=float)
    B = np.array([[5,6],[7,8]], dtype=float)
    assert np.allclose(strassen(A, B), A @ B), \"❌ Error en Strassen\"
    print(\"✅ Validación correcta\\n\")
    run_benchmark()
