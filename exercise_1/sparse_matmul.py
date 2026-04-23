import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import urllib.request
import tarfile
import os
import time

def download_and_load_matrix(url, filename, extract_dir="sparse_data"):
    os.makedirs(extract_dir, exist_ok=True)
    tar_path = os.path.join(extract_dir, filename)
    
    if not os.path.exists(tar_path):
        print(f"Descargando {filename}...")
        try:
            urllib.request.urlretrieve(url, tar_path)
        except Exception as e:
            print(f"Error descargando {filename}: {e}")
            return None
            
    # Extraer el archivo .tar.gz
    mtx_file = ""
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
            for member in tar.getmembers():
                if member.name.endswith(".mtx"):
                    mtx_file = os.path.join(extract_dir, member.name)
                    break
    except Exception as e:
        print(f"Error extrayendo {filename}: {e}")
        return None
        
    if not mtx_file or not os.path.exists(mtx_file):
        print(f"No se encontró el archivo .mtx para {filename}")
        return None
        
    print(f"Cargando {mtx_file}...")
    matrix = sio.mmread(mtx_file)
    return matrix.tocsr()

def run_sparse_benchmark():
    print("=== Ejercicio 1: Multiplicación de Matrices Ralas (Sparse) ===")
    print("Descargando matrices de SuiteSparse Matrix Collection...\n")
    
    # Matriz 1: bcsstk01 (Matriz de rigidez estructural pequeña)
    url1 = "https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk01.tar.gz"
    
    # Matriz 2: gr_30_30 (Matriz de grafos)
    url2 = "https://suitesparse-collection-website.herokuapp.com/MM/HB/gr_30_30.tar.gz"
    
    matrices = [
        ("bcsstk01.tar.gz", url1),
        ("gr_30_30.tar.gz", url2)
    ]
    
    for filename, url in matrices:
        A_sparse = download_and_load_matrix(url, filename)
        
        if A_sparse is None:
            continue
            
        print(f"\\nMatriz: {filename}")
        print(f"Dimensiones: {A_sparse.shape}")
        print(f"Elementos no nulos (NNZ): {A_sparse.nnz}")
        sparsity = 1.0 - (A_sparse.nnz / (A_sparse.shape[0] * A_sparse.shape[1]))
        print(f"Dispersión (Sparsity): {sparsity:.4%}")
        
        # 1. Multiplicación Sparse
        t0 = time.perf_counter()
        # Elevamos al cuadrado la matriz (A x A)
        C_sparse = A_sparse.dot(A_sparse)
        t_sparse = time.perf_counter() - t0
        print(f"Tiempo de multiplicación Sparse (A x A): {t_sparse:.6f} s")
        
        # 2. Multiplicación Densa
        A_dense = A_sparse.toarray()
        t0 = time.perf_counter()
        C_dense = A_dense @ A_dense
        t_dense = time.perf_counter() - t0
        print(f"Tiempo de multiplicación Densa (A x A):  {t_dense:.6f} s")
        
        # 3. Comparativa
        speedup = t_dense / t_sparse if t_sparse > 0 else float('inf')
        print(f"Speedup (Densa / Sparse): {speedup:.2f}x")
        print("-" * 50)
        
    print("\\nConclusión: La multiplicación de matrices ralas utilizando estructuras ")
    print("especializadas (como CSR en SciPy) es órdenes de magnitud más rápida y ")
    print("eficiente en memoria que operar con las mismas matrices en formato denso, ")
    print("especialmente a medida que la matriz crece y la dispersión (sparsity) se ")
    print("acerca a 1 (100%).")

if __name__ == "__main__":
    run_sparse_benchmark()
