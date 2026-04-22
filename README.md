# HPC Unit 3 — Final Assignment

## What is this project?

This repository contains the final assignment for the High Performance Computing (HPC) Unit 3 course. The goal is to implement and compare **serial vs parallel** solutions for four classic scientific computing problems, and to analyze when parallelism actually helps — and when it doesn't.

All experiments were run on **Google Colab** using Python `multiprocessing` and `mpi4py` (MPI).

---

## Exercise 1 — Parallel Matrix Multiplication

Matrix multiplication (`C = A × B`) is one of the most fundamental operations in scientific computing and machine learning. In this exercise we asked: *can we make it faster by splitting the work across multiple CPU cores?*

We implemented five strategies:

- **Serial baseline**: standard numpy multiplication (`A @ B`) as the reference.
- **Row partition**: split matrix A by rows, each worker multiplies its chunk by B.
- **Column partition**: split matrix B by columns, each worker multiplies A by its chunk.
- **Block partition**: divide both matrices into 2D blocks and assign each block to a worker.
- **MPI version**: distribute rows of A across MPI processes using `scatter/gather` collective communication.
- **Strassen**: recursive algorithm that reduces multiplications from 8 to 7 per recursion level, lowering theoretical complexity from O(n³) to ~O(n^2.81).

**What we found:** For these matrix sizes, numpy's built-in `@` operator is already internally parallelized with BLAS/LAPACK, so Python-level multiprocessing actually makes things *slower* due to process spawning overhead. MPI achieved a real speedup of ~2x at n=1024 with 2 processes. Strassen was only faster than numpy for very small matrices (n=128).

---

## Exercise 2 — Parallel Cell Image Processing

Modern biology microscopy pipelines need to analyze hundreds or thousands of images automatically. In this exercise we built an automated pipeline to detect and measure cells in microscopy images.

**Dataset:** DIC-C2DH-HeLa from the Cell Tracking Challenge — 84 grayscale images (512×512 px) of HeLa cancer cells filmed under a microscope.

The pipeline does the following for each image:
1. **Smooth** the image with a Gaussian filter to reduce noise.
2. **Segment** cells from background using Otsu's threshold.
3. **Separate** cells that are touching using the Watershed algorithm.
4. **Measure** each detected cell: area, bounding box, major axis length, minor axis length.

We then parallelized this pipeline using Python `multiprocessing`, distributing images across workers so multiple images are processed simultaneously.

**What we found:** The parallel version with 2 workers achieved a speedup of 1.24x over serial (15.15s → 12.21s for 84 images). On average, each image contained ~52 cells. Interestingly, frames t058–t063 had very few but enormous cells — likely cells in the middle of mitosis (cell division).

---

## Exercise 3 — Forest Fire Cellular Automaton with NASA FIRMS Data

A cellular automaton is a grid-based simulation where each cell changes state based on its neighbors. Forest fire propagation is a natural fit: fire spreads from burning cells to neighboring vegetation.

In this exercise we combined a cellular automaton with **real satellite fire data** from NASA FIRMS (Fire Information for Resource Management System).

**Data:** 5,632 fire hotspot detections over the Yucatan Peninsula (April 2024) from the VIIRS S-NPP satellite sensor. Each detection includes GPS coordinates and Fire Radiative Power (FRP), a measure of fire intensity.

The simulation works as follows:
- Build a 200×200 grid covering the Yucatan Peninsula (~500m per cell).
- Map NASA FIRMS detections onto the grid as initial ignition points (state = *burning*).
- At each time step, burning cells can ignite susceptible neighbors based on a probability that depends on the number of burning neighbors and local FRP intensity.
- Burning cells transition to *burned* after 3 steps (fuel exhausted).

We ran 20 simulation steps and parallelized using MPI with domain decomposition: each process handles a horizontal strip of the grid and exchanges boundary rows (halos) with neighbors at each step.

**What we found:** The fire spread rapidly, consuming most of the central Yucatan region by step 20. MPI with 4 processes achieved 1.34x speedup over serial. For small grids, halo exchange communication limits scalability. An important note: NASA FIRMS hotspots are thermal anomaly detections from satellite — they are not the same as the actual fire perimeter.

---

## Exercise 4 — Parallel K-Means Clustering

*In progress.*

---

## Repository Structure

```
hpc-unit3/
├── README.md
├── requirements.txt
├── setup_colab.py          ← run this first on Google Colab
│
├── exercise_1/             ← Matrix Multiplication 
│   ├── serial_matmul.py
│   ├── parallel_row.py
│   ├── parallel_col.py
│   ├── parallel_block.py
│   ├── mpi_matmul.py
│   └── strassen.py
│
├── exercise_2/             ← Cell Image Processing 
│   ├── serial_pipeline.py
│   ├── parallel_pipeline.py
│   └── results/
│       └── exercise_2_results.csv
│
├── exercise_3/             ← Forest Fire Automaton 
│   ├── serial_automaton.py
│   ├── mpi_automaton.py
│   ├── firms_data.csv
│   └── frames/
│
├── exercise_4/             ← Parallel K-Means 
│   ├── serial_kmeans.py
│   └── mpi_kmeans.py
│
└── docs/
    ├── report.pdf
    └── assets/
```

## How to run

### Setup (Google Colab)
```python
# Cell 1 — install system MPI
!apt-get install -y -q libopenmpi-dev openmpi-bin

# Cell 2 — install Python dependencies
!pip install -q numpy scipy matplotlib pandas seaborn mpi4py \
    cellpose scikit-image opencv-python-headless Pillow tqdm \
    requests geopandas shapely pyproj imageio scikit-learn ucimlrepo psutil
```

### Exercise 1
```bash
python exercise_1/serial_matmul.py
python exercise_1/parallel_row.py
python exercise_1/parallel_col.py
python exercise_1/parallel_block.py
mpirun --allow-run-as-root --oversubscribe -n 4 python exercise_1/mpi_matmul.py
python exercise_1/strassen.py
```

### Exercise 2
```bash
python exercise_2/serial_pipeline.py
python exercise_2/parallel_pipeline.py
```

### Exercise 3
```bash
python exercise_3/serial_automaton.py
mpirun --allow-run-as-root --oversubscribe -n 4 python exercise_3/mpi_automaton.py
```

### Exercise 4
```bash
python exercise_4/serial_kmeans.py
mpirun --allow-run-as-root --oversubscribe -n 4 python exercise_4/mpi_kmeans.py
```

## Execution Environment

| Item     | Value                                               |
|----------|-----------------------------------------------------|
| Platform | Google Colab                                        |
| Python   | 3.12                                                |
| MPI      | OpenMPI 4.x (2 physical CPUs, oversubscribed to 4)  |
| numpy    | 2.0.2                                               |
| scipy    | 1.16.3                                              |
| mpi4py   | 4.1.1                                               |
| sklearn  | 1.6.1                                               |

## Authors
-Acosta Castellanos Bianca 
-Canche Chuc Angel Rivaldo
-Ku Russel
-Sanchez Novelo Damian
-Velasco Jonathan
