# Parallel-Programming-Applications-2.0
# HPC Unit 3 — Final Assignment

## Objective
This repository contains the serial and parallel implementations for the four exercises of the High Performance Computing Unit 3 Final Assignment. Each exercise includes a serial baseline and at least one parallel version using Python `multiprocessing` and/or `mpi4py`.

## Repository Structure

```
hpc-unit3/
├── README.md
├── requirements.txt
├── setup_colab.py          ← run this first on Google Colab
│
├── exercise_1/             ← Parallel Matrix Multiplication
│   ├── serial_matmul.py
│   ├── parallel_row.py
│   ├── parallel_col.py
│   ├── parallel_block.py
│   ├── mpi_matmul.py
│   ├── strassen.py
│   └── benchmark.py
│
├── exercise_2/             ← Cell Image Processing
│   ├── serial_pipeline.py
│   ├── parallel_pipeline.py
│   └── results/
│
├── exercise_3/             ← Forest Fire Automaton (NASA FIRMS)
│   ├── fetch_firms_data.py
│   ├── serial_automaton.py
│   ├── mpi_automaton.py
│   └── visualize.py
│
├── exercise_4/             ← Parallel K-Means (Covertype)
│   ├── serial_kmeans.py
│   ├── mpi_kmeans.py
│   └── benchmark.py
│
└── docs/
    ├── report.pdf
    └── assets/
```

## Software Requirements

- Python >= 3.10
- OpenMPI >= 4.x (system library, required for `mpi4py`)
- See `requirements.txt` for all Python dependencies

### Google Colab (recommended)
```python
# Run at the start of each session:
exec(open('setup_colab.py').read())
```

# Install Python dependencies
pip install -r requirements.txt
```

## Running the Exercises

### Exercise 1 — Matrix Multiplication
```bash
# Serial baseline
python exercise_1/serial_matmul.py

# Parallel (multiprocessing, row partition)
python exercise_1/parallel_row.py

# MPI (4 processes)
mpirun -n 4 python exercise_1/mpi_matmul.py

# Full benchmark
python exercise_1/benchmark.py
```

### Exercise 2 — Cell Image Processing
```bash
python exercise_2/serial_pipeline.py
python exercise_2/parallel_pipeline.py --workers 4
```

### Exercise 3 — Forest Fire Automaton
```bash
python exercise_3/fetch_firms_data.py   # download NASA FIRMS data
python exercise_3/serial_automaton.py
mpirun -n 4 python exercise_3/mpi_automaton.py
```

### Exercise 4 — K-Means Clustering
```bash
python exercise_4/serial_kmeans.py
mpirun -n 4 python exercise_4/mpi_kmeans.py
```

## Execution Environment

| Item | Value |
|------|-------|
| Platform | Google Colab / Linux |
| Python | 3.10+ |
| MPI | OpenMPI 4.x |
| CPU | (document here after running) |
| RAM | (document here after running) |

## Authors
-Acosta Castellanos Bianca 
-Canche Chuc Angel Rivaldo
-Ku Russel
-Sanchez Novelo Damian
-Velasco Jonathan
