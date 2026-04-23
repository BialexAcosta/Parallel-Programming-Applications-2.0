import numpy as np
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def run(n):
    if rank == 0:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        t_serial_start = time.perf_counter()
        _ = A @ B
        t_serial = time.perf_counter() - t_serial_start
    else:
        A = None
        B = None
        t_serial = None

    # Broadcast B a todos los procesos
    B = comm.bcast(B, root=0)

    # Dividir filas de A entre procesos
    if rank == 0:
        row_chunks = np.array_split(A, size, axis=0)
    else:
        row_chunks = None

    local_A = comm.scatter(row_chunks, root=0)

    # Cada proceso multiplica su bloque
    t0 = MPI.Wtime()
    local_C = local_A @ B
    comm.Barrier()
    t_par = MPI.Wtime() - t0

    # Reunir resultados
    C_parts = comm.gather(local_C, root=0)

    if rank == 0:
        C = np.vstack(C_parts)
        print(f"n={n:>5}  procs={size}  T_serial={t_serial:.4f}s  T_mpi={t_par:.4f}s  Speedup={t_serial/t_par:.2f}  Eficiencia={t_serial/t_par/size:.2f}")

if __name__ == "__main__":
    for n in [128, 256, 512, 1024]:
        run(n)
