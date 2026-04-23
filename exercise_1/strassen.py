import numpy as np
import time

def strassen(A, B):
    n = A.shape[0]
    # Caso base: Para matrices pequeñas, el producto de NumPy (basado en BLAS) es imbatible
    if n <= 64:
        return A @ B

    mid = n // 2
    # División de matrices en cuadrantes
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]

    # Las 7 multiplicaciones recursivas de Strassen
    M1 = strassen(A11 + A22, B11 + B22)
    M2 = strassen(A21 + A22, B11)
    M3 = strassen(A11,        B12 - B22)
    M4 = strassen(A22,        B21 - B11)
    M5 = strassen(A11 + A12, B22)
    M6 = strassen(A21 - A11, B11 + B12)
    M7 = strassen(A12 - A22, B21 + B22)

    # Reconstrucción de la matriz resultante
    C = np.zeros((n, n))
    C[:mid, :mid] = M1 + M4 - M5 + M7
    C[:mid, mid:] = M3 + M5
    C[mid:, :mid] = M2 + M4
    C[mid:, mid:] = M1 - M2 + M3 + M6
    return C

def pad_to_power_of_2(A):
    """Añade ceros a la matriz para que su dimensión sea potencia de 2."""
    n = max(A.shape)
    p = 1
    while p < n:
        p *= 2
    padded = np.zeros((p, p))
    padded[:A.shape[0], :A.shape[1]] = A
    return padded

def run_benchmark(sizes=[128, 256, 512]):
    print(f"{'Tamaño':>8} {'T serial':>10} {'T strassen':>12} {'Speedup':>9}")
    print("-" * 45)
    for n in sizes:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        # Preparación para Strassen (potencia de 2)
        Ap = pad_to_power_of_2(A)
        Bp = pad_to_power_of_2(B)

        # Medición Serial (NumPy optimizado)
        t0 = time.perf_counter()
        C_serial = A @ B
        t_serial = time.perf_counter() - t0

        # Medición Strassen
        t0 = time.perf_counter()
        C_strassen = strassen(Ap, Bp)[:n, :n]
        t_strassen = time.perf_counter() - t0

        # Validación técnica
        assert np.allclose(C_serial, C_strassen, atol=1e-6), "Error en la validación"
        
        speedup = t_serial / t_strassen
        print(f"{n:<8} {t_serial:>10.4f} {t_strassen:>12.4f} {speedup:>9.2f}")

if __name__ == "__main__":
    # Prueba inicial de validación
    A_test = np.array([[1, 2], [3, 4]], dtype=float)
    B_test = np.array([[5, 6], [7, 8]], dtype=float)
    
    if np.allclose(strassen(A_test, B_test), A_test @ B_test):
        print("✅ Validación inicial correcta\n")
        run_benchmark()
    else:
        print("❌ Error en la lógica de Strassen")