import time
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

def load_data():
    print("Descargando Covertype dataset (puede tomar unos segundos)...")
    covertype = fetch_ucirepo(id=31)
    X = covertype.data.features.values
    # Tomamos una muestra representativa de 100k filas para que corra más rápido
    # El dataset completo tiene >500k filas, lo cual es excelente para probar paralelismo
    print(f"Dataset original cargado. Forma: {X.shape}")
    X = X[:100000]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def kmeans_serial(X, k=7, max_iters=20, tol=1e-4):
    n_samples, n_features = X.shape
    # Inicializar centroides al azar
    np.random.seed(42)
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices]
    
    t0 = time.perf_counter()
    for i in range(max_iters):
        # 1. Asignación (Distancia euclidiana)
        # Broadcasting para encontrar la distancia desde cada punto a cada centroide
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # 2. Actualización de centroides
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[j] = centroids[j] # si el cluster está vacío
                
        # Verificar convergencia
        diff = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if diff < tol:
            print(f"Convergencia alcanzada en la iteración {i+1}")
            break
            
    t_elapsed = time.perf_counter() - t0
    return centroids, labels, t_elapsed

if __name__ == "__main__":
    print("=== Ejercicio 4: K-Means Clustering (Serial) ===")
    X = load_data()
    print("Ejecutando K-Means Serial (K=7)...")
    centroids, labels, t_serial = kmeans_serial(X, k=7, max_iters=20)
    print(f"Tiempo K-Means Serial (100k filas): {t_serial:.4f}s")