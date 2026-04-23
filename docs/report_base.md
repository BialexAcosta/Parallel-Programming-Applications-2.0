# HPC Unit 3 - Final Assignment Report 

## Execution Environment

- Operating system:
- Python version:
- MPI runtime and version:
- CPU and RAM:
- Relevant package versions:

## Reproducibility Settings

- Random seed policy:
- Dataset versions/sources:
- Preprocessing assumptions:
- Execution command templates:

## Exercise 1. Parallel Matrix Multiplication

### 1. Evidence of Completeness for Each Task

- Serial baseline implemented in `exercise_1/serial_matmul.py`.
- Multiprocessing row partition implemented in `exercise_1/parallel_row.py`.
- Multiprocessing column partition implemented in `exercise_1/parallel_col.py`.
- Block or quadrant decomposition implemented in `exercise_1/parallel_block.py`.
- Distributed-memory MPI version implemented in `exercise_1/mpi_matmul.py`.
- Strassen-based method implemented in `exercise_1/strassen.py`.
- Sparse matrix experiments implemented in `exercise_1/sparse_matmul.py` using SuiteSparse matrices.
- Insert code snippets, benchmark logs, and validation outputs here.

### 2. Results

- Add runtime tables for serial vs parallel methods over increasing dense matrix sizes.
- Add speedup and efficiency values for each worker/process configuration.
- Include sparse-matrix comparison (sparse vs dense representation runtime).
- Insert plots or tables with clear labels.

### 3. Discussion

- Explain bottlenecks in each strategy.
- Discuss communication, memory movement, and load balance.
- Explain when Strassen is beneficial or not.
- Explain how sparsity pattern changes observed performance.

## Exercise 2. Parallel Cell Image Processing and Morphological Characterization

### 1. Evidence of Completeness for Each Task

- Dataset ingestion and inspection performed using DIC-C2DH-HeLa under `exercise_2/DIC-C2DH-HeLa`.
- Serial segmentation and measurement pipeline implemented in `exercise_2/serial_pipeline.py`.
- Multiprocessing pipeline implemented in `exercise_2/parallel_pipeline.py`.
- Optional pretrained model integration implemented in `exercise_2/cellpose_pipeline.py`.
- Summary outputs stored in `exercise_2/results`.
- Insert sample segmentations, bounding boxes, and per-image output examples.

### 2. Results

- Add serial vs parallel timing for multiple worker counts.
- Add per-image summary table with number of cells, average width, average length, and variability.
- State clearly whether measurements are in pixels, micrometers, or both.

### 3. Discussion

- Discuss segmentation quality limitations and their effect on geometry metrics.
- Compare watershed and pretrained-model behavior where applicable.
- Describe key error modes and tradeoffs.

## Exercise 3. Forest Fire Cellular Automaton Driven by NASA FIRMS Data

### 1. Evidence of Completeness for Each Task

- FIRMS data acquisition and filtering implemented in `download_firms.py`.
- Grid mapping and serial automaton implemented in `exercise_3/serial_automaton.py`.
- MPI domain decomposition and halo exchange implemented in `exercise_3/mpi_automaton.py`.
- Visual output example saved in `exercise_3/frames/final_state.png`.
- Insert snapshots/animation references and selected logs.

### 2. Results

- Add runtime comparison across grid sizes and/or simulation horizons.
- Add process-scaling table for MPI runs.
- Include representative temporal evolution figures.

### 3. Discussion

- Explain interpretation limits of hotspot detections relative to true perimeter.
- Discuss effects of simplified local ignition and burn-lifetime assumptions.
- Identify communication overhead and decomposition constraints.

## Exercise 4. Parallel K-Means Clustering

### 1. Evidence of Completeness for Each Task

- Serial K-means baseline implemented in `exercise_4/serial_kmeans.py`.
- MPI K-means implementation implemented in `exercise_4/mpi_kmeans.py`.
- Dataset loading and preprocessing performed on Covertype dataset.
- Insert assignment/update logic snippets and MPI reduction evidence.

### 2. Results

- Add per-iteration and total runtime comparisons.
- Add scaling comparisons for different process counts.
- Add experiments for different `k` values.
- Include convergence behavior and clustering stability notes.

### 3. Discussion

- Describe when parallel K-means becomes advantageous.
- Analyze collective communication costs.
- Discuss influence of dataset size and number of clusters on scalability.

- All figures and tables are readable and correctly labeled.
- Serial and parallel comparisons are easy to verify.
- Command lines and execution settings are documented.
- Design choices are justified with technical reasoning.
