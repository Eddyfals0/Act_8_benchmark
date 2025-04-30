import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, float32
import time

# Kernel de multiplicación de matrices para GPU
@cuda.jit
def matrix_mul_gpu_kernel(A, B, C, N):
    row, col = cuda.grid(2)
    if row < N and col < N:
        tmp = 0.0
        for k in range(N):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

def matrix_mul_cpu(A, B):
    return np.dot(A, B)

def matrix_mul_gpu(A, B):
    N = A.shape[0]
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)

    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array((N, N), dtype=np.float32)

    threadsperblock = (16, 16)
    blockspergrid_x = (N + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (N + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    matrix_mul_gpu_kernel[blockspergrid, threadsperblock](d_A, d_B, d_C, N)
    cuda.synchronize()
    return d_C.copy_to_host()

def benchmark(sizes, runs=5):
    cpu_times = []
    gpu_times = []

    for N in sizes:
        print(f"Ejecutando benchmark para tamaño {N}x{N}")
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)

        # --- CPU ---
        cpu_run_times = []
        for _ in range(runs):
            start = time.perf_counter()
            C_cpu = matrix_mul_cpu(A, B)
            end = time.perf_counter()
            cpu_run_times.append(end - start)
        avg_cpu_time = sum(cpu_run_times) / runs
        print("  CPU:", " ".join([f"{t:.4f}s" for t in cpu_run_times]), f"(promedio: {avg_cpu_time:.4f}s)")

        # --- GPU ---
        gpu_run_times = []
        try:
            for _ in range(runs):
                start = time.perf_counter()
                C_gpu = matrix_mul_gpu(A, B)
                end = time.perf_counter()
                gpu_run_times.append(end - start)
            avg_gpu_time = sum(gpu_run_times) / runs
            error = np.max(np.abs(C_cpu - C_gpu))
            print("  GPU:", " ".join([f"{t:.4f}s" for t in gpu_run_times]), f"(promedio: {avg_gpu_time:.4f}s)")
            print(f"  Error máximo: {error}")
        except cuda.CudaAPIError as e:
            print("Error de CUDA (posible timeout o memoria):", e)
            avg_gpu_time = None

        cpu_times.append(avg_cpu_time)
        gpu_times.append(avg_gpu_time)

    return cpu_times, gpu_times

def plot_results(sizes, cpu_times, gpu_times):
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cpu_times, 'o-', label='CPU')
    plt.plot(sizes, gpu_times, 's-', label='GPU')
    plt.xlabel('Tamaño de matriz (NxN)')
    plt.ylabel('Tiempo promedio (s)')
    plt.title('Benchmark CPU vs GPU')
    plt.legend()
    plt.grid(True)
    plt.savefig("benchmark_results.png")
    plt.show()

if __name__ == "__main__":
    sizes = [128, 256, 512, 1024]  
    cpu_times, gpu_times = benchmark(sizes)
    plot_results(sizes, cpu_times, gpu_times)
