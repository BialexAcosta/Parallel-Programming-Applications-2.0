# Reporte Final - HPC Unit 3

## Ejercicio 1: Multiplicación de Matrices
### Tareas Completadas
Se implementaron las versiones serial, por filas, por columnas, bloques, MPI y Strassen (presentes en el Notebook original). Adicionalmente, se completó la prueba con **matrices ralas (sparse)** de la colección SuiteSparse (`bcsstk01` y `gr_30_30`).

### Resultados y Discusión
La multiplicación de matrices ralas demostró que para matrices con alta dispersión (sparsity > 90%), utilizar estructuras de datos especializadas (como CSR en SciPy) es varios órdenes de magnitud más rápido que la multiplicación densa tradicional, debido a que se omiten las operaciones multiplicativas con ceros y se reduce drásticamente el consumo de memoria. En cuanto al rendimiento en paralelo (multiprocessing) de matrices densas, se observó que es inferior al serial puro de NumPy porque internamente NumPy ya está paralelizado y optimizado (BLAS/LAPACK), causando que la creación de procesos en Python añada una sobrecarga innecesaria.

---

## Ejercicio 2: Procesamiento de Imágenes
### Tareas Completadas
Se completó el pipeline de segmentación original (Watershed) y se añadió la implementación de **Cellpose** (`cellpose_pipeline.py`), que utiliza modelos pre-entrenados para mejorar la calidad de segmentación, especialmente en las células en mitosis.

### Resultados y Discusión
Se evaluaron imágenes del dataset DIC-C2DH-HeLa. Cellpose ("cyto2") permite identificar mejor los bordes sin necesidad de aplicar umbrales manuales complejos, obteniendo contornos más naturales. Se procesaron parámetros como el área, y las longitudes del eje mayor y menor para cada célula detectada, exportando los resultados a formato CSV. 

---

## Ejercicio 3: Autómata Celular de Incendios Forestales
### Tareas Completadas
Se extrajeron los datos de puntos de calor de NASA FIRMS sobre Yucatán y se implementó un autómata celular tanto serial como paralelo usando MPI. 

### Resultados y Discusión
En la versión MPI, el área de simulación (grid) se dividió en franjas horizontales (descomposición de dominio) y se intercambiaron halos (fronteras) mediante `Isend` e `Irecv` para mantener sincronizado el estado. Se observó que los puntos de detección satelital (FRP) son útiles para iniciar los focos de incendio en la simulación; sin embargo, no equivalen al perímetro exacto del incendio real, debido a la resolución del satélite y las nubes. 

---

## Ejercicio 4: Clustering Paralelo K-Means
### Tareas Completadas
Se implementó K-Means de manera serial y con paso de mensajes MPI evaluando sobre el Covertype Dataset.

### Resultados y Discusión
Los datos se distribuyeron (Scatterv) a todos los procesos. En cada iteración, cada proceso calcula los centroides localmente, seguidos por un `Allreduce` para sumar coordenadas y tamaños de cluster globalmente, actualizando así el centroide real. Para grandes cantidades de datos (como las 500k muestras de Covertype), la versión distribuida con MPI demuestra ventajas claras en tiempo de procesamiento en contraste con la serial. El principal cuello de botella en MPI es el `Allreduce` si la cantidad de clusters `k` es muy grande, pero para `k=7` (el número de clases en Covertype), el overhead de comunicación es mínimo comparado con el trabajo de cálculo de distancia euclidiana.
