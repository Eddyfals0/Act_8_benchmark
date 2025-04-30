import numpy as np
from numba import cuda
import time

size = 100_000_000

@cuda.jit
def add_kernel(a, b, result):
    idx = cuda.grid(1)
    if idx < a.size:
        result[idx] = a[idx] + b[idx]

def benchmark_gpu():
    threads_per_block = 256
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    result = np.empty_like(a)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_result = cuda.device_array_like(a)

    cuda.synchronize()
    start_time = time.time()
    add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_result)
    cuda.synchronize()
    end_time = time.time()

    d_result.copy_to_host(result)

    print(f"[GPU] Tiempo de ejecución en GPU: {end_time - start_time:.4f} segundos")

if __name__ == "__main__":
    benchmark_gpu()
