from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

vector_size = 100_000_000

# Cada proceso genera su parte de los vectores
chunk_size = vector_size // size
start = rank * chunk_size
end = vector_size if rank == size - 1 else (rank + 1) * chunk_size

# Solo el proceso raíz mide el tiempo
if rank == 0:
    global_start_time = time.time()

# Todos los procesos generan su propio segmento de datos
a_chunk = np.random.rand(end - start).astype(np.float32)
b_chunk = np.random.rand(end - start).astype(np.float32)

# Suma local
local_result = a_chunk + b_chunk

# El proceso raíz recopila todos los resultados
if rank == 0:
    result = np.empty(vector_size, dtype=np.float32)
else:
    result = None

comm.Gather(local_result, result, root=0)

if rank == 0:
    global_end_time = time.time()
    print(f"[CPU-MPI] Tiempo de ejecución con {size} procesos: {global_end_time - global_start_time:.4f} segundos")
