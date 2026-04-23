import time
import numpy as np
from mpi4py import MPI
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

def mpi_kmeans():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    k = 7
    max_iters = 20
    tol = 1e-4
    
    if rank == 0:
        print("=== Ejercicio 4: K-Means Clustering (MPI) ===")
        print("Cargando Covertype dataset...")
        covertype = fetch_ucirepo(id=31)
        X = covertype.data.features.values
        X = X[:100000] # Tomamos una muestra de 100k para benchmarking
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_samples, n_features = X_scaled.shape
        print(f"Dataset escalado y listo. Forma: {X_scaled.shape}")
        
        # Inicializar centroides
        np.random.seed(42)
        indices = np.random.choice(n_samples, k, replace=False)
        centroids = X_scaled[indices]
    else:
        n_samples, n_features = None, None
        X_scaled = None
        centroids = None
        
    # Compartir metadata
    n_samples = comm.bcast(n_samples, root=0)
    n_features = comm.bcast(n_features, root=0)
    
    if rank != 0:
        centroids = np.zeros((k, n_features))
        
    centroids = comm.bcast(centroids, root=0)
    
    # Dividir el trabajo (filas de X)
    chunk_size = n_samples // size
    counts = [chunk_size] * size
    counts[-1] += n_samples % size # El último proceso toma el resto
    
    displacements = [sum(counts[:i]) for i in range(size)]
    
    local_count = counts[rank]
    local_X = np.zeros((local_count, n_features))
    
    # Distribuir datos (Scatterv)
    comm.Scatterv([X_scaled, counts, displacements, MPI.DOUBLE], local_X, root=0)
    
    comm.Barrier()
    t0 = MPI.Wtime()
    
    for it in range(max_iters):
        # 1. Asignación local
        distances = np.linalg.norm(local_X[:, np.newaxis, :] - centroids, axis=2)
        local_labels = np.argmin(distances, axis=1)
        
        # 2. Sumas y conteos locales
        local_sums = np.zeros((k, n_features))
        local_counts = np.zeros(k)
        
        for j in range(k):
            cluster_points = local_X[local_labels == j]
            local_counts[j] = len(cluster_points)
            if len(cluster_points) > 0:
                local_sums[j] = np.sum(cluster_points, axis=0)
                
        # 3. Reducción global
        global_sums = np.zeros_like(local_sums)
        global_counts = np.zeros_like(local_counts)
        
        comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
        comm.Allreduce(local_counts, global_counts, op=MPI.SUM)
        
        # 4. Actualización de centroides en todos los procesos
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            if global_counts[j] > 0:
                new_centroids[j] = global_sums[j] / global_counts[j]
            else:
                new_centroids[j] = centroids[j]
                
        diff = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        
        if diff < tol:
            if rank == 0:
                print(f"Convergencia alcanzada en la iteración {it+1}")
            break
            
    comm.Barrier()
    t_par = MPI.Wtime() - t0
    
    if rank == 0:
        print(f"Tiempo K-Means MPI (procesos={size}): {t_par:.4f}s")
        
if __name__ == "__main__":
    mpi_kmeans()