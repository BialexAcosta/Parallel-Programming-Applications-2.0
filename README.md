# High Performance Computing - Unit 3 Final Assignment

Team:
Rivaldo Canche
Bianca Acosta
Russel Ku
Damian Novelo
Jonathan Velasco 
## Assignment Objective

This repository contains four High Performance Computing exercises focused on scientific computing, data science, and AI workflows.

Each exercise includes:
- a serial baseline
- at least one parallel implementation using `multiprocessing`, `mpi4py`, or both
- performance-oriented execution with runtime comparison

The goal is to show, through reproducible experiments, when parallelization improves performance and when overheads limit speedup.

## Repository Structure

```text
Parallel-Programming-Applications-2.0/
├── README.md
├── requirements.txt
├── setup_colab.py
├── download_data.py
├── download_firms.py
│
├── exercise_1/
│   ├── serial_matmul.py
│   ├── parallel_row.py
│   ├── parallel_col.py
│   ├── parallel_block.py
│   ├── mpi_matmul.py
│   ├── strassen.py
│   └── sparse_matmul.py
│
├── exercise_2/
│   ├── serial_pipeline.py
│   ├── parallel_pipeline.py
│   ├── cellpose_pipeline.py
│   ├── DIC-C2DH-HeLa/
│   └── results/
│
├── exercise_3/
│   ├── serial_automaton.py
│   ├── mpi_automaton.py
│   ├── firms_data.csv
│   └── frames/
│
├── exercise_4/
│   ├── serial_kmeans.py
│   └── mpi_kmeans.py
│
└── docs/
      └── report_base.md
```

## Software Requirements

- Python 3.8+
- MPI runtime for distributed scripts
   - Linux/macOS: OpenMPI (`mpirun`)
   - Windows: MS-MPI (`mpiexec`)
- Python dependencies from `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproducibility Notes

- Exercise scripts are organized so serial and parallel versions are easy to compare.
- Benchmark outputs include runtime and, where implemented, speedup and efficiency.
- For fair CPU baseline comparison in Exercise 1, BLAS threading is constrained in matrix scripts.
- Any missing runtime support in a local machine (for example MPI runtime) must be installed before running MPI exercises.

## Data Preparation

Run once before the related exercises:

```bash
python download_data.py
python download_firms.py
```

## How To Run Each Exercise

Use `python` for serial/multiprocessing scripts and `mpirun` or `mpiexec` for MPI scripts.

Exercise 1:

```bash
python exercise_1/serial_matmul.py
python exercise_1/parallel_row.py
python exercise_1/parallel_col.py
python exercise_1/parallel_block.py
python exercise_1/strassen.py
python exercise_1/sparse_matmul.py
mpiexec -n 4 python exercise_1/mpi_matmul.py
```

Exercise 2:

```bash
python exercise_2/serial_pipeline.py
python exercise_2/parallel_pipeline.py
python exercise_2/cellpose_pipeline.py
```

Exercise 3:

```bash
python exercise_3/serial_automaton.py
mpiexec -n 4 python exercise_3/mpi_automaton.py
```

Exercise 4:

```bash
python exercise_4/serial_kmeans.py
mpiexec -n 4 python exercise_4/mpi_kmeans.py
```
