import numpy as np
import time
from numba import cuda
import math
import warnings
import gc
from numba.cuda.cudadrv.devicearray import NumbaPerformanceWarning

# Suprimir advertencias de Numba
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

# Configuración para GPU
BLOCK_SIZE = 16  # Tamaño de bloque para kernels CUDA

# Kernel CUDA para multiplicación de matrices estándar
@cuda.jit
def multiply_matrices_cuda(A, B, C):
    """Kernel CUDA para multiplicación de matrices tradicional"""
    row, col = cuda.grid(2)
    
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

# Función para realizar multiplicación de matrices estándar en GPU
def matrix_mul_gpu_standard(A, B):
    """Multiplicación de matrices estándar en GPU"""
    A = np.ascontiguousarray(A)  # Asegurar que A sea contigua
    B = np.ascontiguousarray(B)  # Asegurar que B sea contigua
    
    n = A.shape[0]
    m = B.shape[1]
    k = A.shape[1]
    
    # Asignar memoria en la GPU
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array((n, m), np.float32)
    
    # Configurar la cuadrícula de threads
    threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
    blocks_per_grid_x = int(math.ceil(n / threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(m / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Lanzar el kernel
    cuda.synchronize()
    start_time = time.perf_counter()
    multiply_matrices_cuda[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
    cuda.synchronize()
    end_time = time.perf_counter()
    
    # Copiar el resultado de vuelta a la CPU
    C = d_C.copy_to_host()
    
    # Liberar memoria GPU
    del d_A
    del d_B
    del d_C
    
    return C, end_time - start_time

# Kernel CUDA para sumar matrices
@cuda.jit
def add_matrices_cuda(A, B, C):
    """Kernel CUDA para sumar matrices"""
    row, col = cuda.grid(2)
    
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = A[row, col] + B[row, col]

# Kernel CUDA para restar matrices
@cuda.jit
def subtract_matrices_cuda(A, B, C):
    """Kernel CUDA para restar matrices"""
    row, col = cuda.grid(2)
    
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = A[row, col] - B[row, col]

# Función para sumar matrices en GPU
def add_matrices_gpu(d_A, d_B):
    """Suma matrices en GPU sin transferencias CPU"""
    n = d_A.shape[0]
    m = d_A.shape[1]
    
    d_C = cuda.device_array((n, m), np.float32)
    
    threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
    blocks_per_grid_x = int(math.ceil(n / threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(m / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    add_matrices_cuda[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
    
    return d_C

# Función para restar matrices en GPU
def subtract_matrices_gpu(d_A, d_B):
    """Resta matrices en GPU sin transferencias CPU"""
    n = d_A.shape[0]
    m = d_A.shape[1]
    
    d_C = cuda.device_array((n, m), np.float32)
    
    threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
    blocks_per_grid_x = int(math.ceil(n / threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(m / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    subtract_matrices_cuda[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
    
    return d_C

# Función para realizar Strassen en GPU mejorada
def strassen_gpu_optimized(A, B, threshold=512):
    """Algoritmo de Strassen optimizado para GPU con umbral ajustado"""
    A = np.ascontiguousarray(A)  # Asegurar que A sea contigua
    B = np.ascontiguousarray(B)  # Asegurar que B sea contigua
    
    n = A.shape[0]
    
    # Si la matriz es pequeña, usar multiplicación estándar
    if n <= threshold:
        result, _ = matrix_mul_gpu_standard(A, B)
        return result
    
    # Transferir matrices a GPU una sola vez
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    
    # Usar función recursiva que mantiene datos en GPU
    d_C = _strassen_gpu_recursive(d_A, d_B, threshold)
    
    # Transferir resultado de vuelta a CPU
    C = d_C.copy_to_host()
    
    # Liberar memoria GPU
    del d_A
    del d_B
    del d_C
    
    return C

# Función recursiva que mantiene datos en GPU
def _strassen_gpu_recursive(d_A, d_B, threshold):
    """Implementación recursiva de Strassen que mantiene datos en GPU"""
    n = d_A.shape[0]
    
    # Si la matriz es pequeña, usar multiplicación estándar en GPU
    if n <= threshold:
        n = d_A.shape[0]
        m = d_B.shape[1]
        
        d_C = cuda.device_array((n, m), np.float32)
        
        threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
        blocks_per_grid_x = int(math.ceil(n / threads_per_block[0]))
        blocks_per_grid_y = int(math.ceil(m / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        multiply_matrices_cuda[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
        
        return d_C
    
    # Dividir matrices en cuadrantes (en GPU)
    half_n = n // 2
    
    d_A11 = d_A[:half_n, :half_n]
    d_A12 = d_A[:half_n, half_n:]
    d_A21 = d_A[half_n:, :half_n]
    d_A22 = d_A[half_n:, half_n:]
    
    d_B11 = d_B[:half_n, :half_n]
    d_B12 = d_B[:half_n, half_n:]
    d_B21 = d_B[half_n:, :half_n]
    d_B22 = d_B[half_n:, half_n:]
    
    # Calcular sumas y restas necesarias para P1-P7
    d_A11_A22 = add_matrices_gpu(d_A11, d_A22)
    d_B11_B22 = add_matrices_gpu(d_B11, d_B22)
    
    d_A21_A22 = add_matrices_gpu(d_A21, d_A22)
    
    d_B12_B22 = subtract_matrices_gpu(d_B12, d_B22)
    
    d_B21_B11 = subtract_matrices_gpu(d_B21, d_B11)
    
    d_A11_A12 = add_matrices_gpu(d_A11, d_A12)
    
    d_A21_A11 = subtract_matrices_gpu(d_A21, d_A11)
    d_B11_B12 = add_matrices_gpu(d_B11, d_B12)
    
    d_A12_A22 = subtract_matrices_gpu(d_A12, d_A22)
    d_B21_B22 = add_matrices_gpu(d_B21, d_B22)
    
    # Calcular productos intermedios P1 a P7 secuencialmente
    d_P1 = _strassen_gpu_recursive(d_A11_A22, d_B11_B22, threshold)
    
    # Liberar memoria que ya no se necesita
    del d_A11_A22
    del d_B11_B22
    
    d_P2 = _strassen_gpu_recursive(d_A21_A22, d_B11, threshold)
    
    del d_A21_A22
    
    d_P3 = _strassen_gpu_recursive(d_A11, d_B12_B22, threshold)
    
    del d_B12_B22
    
    d_P4 = _strassen_gpu_recursive(d_A22, d_B21_B11, threshold)
    
    del d_B21_B11
    
    d_P5 = _strassen_gpu_recursive(d_A11_A12, d_B22, threshold)
    
    del d_A11_A12
    
    d_P6 = _strassen_gpu_recursive(d_A21_A11, d_B11_B12, threshold)
    
    del d_A21_A11
    del d_B11_B12
    
    d_P7 = _strassen_gpu_recursive(d_A12_A22, d_B21_B22, threshold)
    
    del d_A12_A22
    del d_B21_B22
    
    # Limpiar referencias que ya no se necesitan
    del d_A11, d_A12, d_A21, d_A22
    del d_B11, d_B12, d_B21, d_B22
    
    # Calcular cuadrantes del resultado en GPU
    d_C11_temp1 = add_matrices_gpu(d_P1, d_P4)
    d_C11_temp2 = subtract_matrices_gpu(d_C11_temp1, d_P5)
    d_C11 = add_matrices_gpu(d_C11_temp2, d_P7)
    
    del d_C11_temp1
    del d_C11_temp2
    
    d_C12 = add_matrices_gpu(d_P3, d_P5)
    
    d_C21 = add_matrices_gpu(d_P2, d_P4)
    
    d_C22_temp1 = subtract_matrices_gpu(d_P1, d_P2)
    d_C22_temp2 = add_matrices_gpu(d_C22_temp1, d_P3)
    d_C22 = add_matrices_gpu(d_C22_temp2, d_P6)
    
    del d_C22_temp1
    del d_C22_temp2
    
    # Liberar memoria de P1-P7 que ya no se necesita
    del d_P1, d_P2, d_P3, d_P4, d_P5, d_P6, d_P7
    
    # Combinar cuadrantes en una sola matriz resultado
    d_C = cuda.device_array((n, n), np.float32)
    
    # Copiar los cuadrantes a la matriz resultado
    d_C[:half_n, :half_n] = d_C11
    d_C[:half_n, half_n:] = d_C12
    d_C[half_n:, :half_n] = d_C21
    d_C[half_n:, half_n:] = d_C22
    
    del d_C11, d_C12, d_C21, d_C22
    
    return d_C

# Función para hacer padding a la matriz
def pad_matrix(matrix):
    """Rellena la matriz para que tenga dimensiones potencia de 2"""
    n = matrix.shape[0]
    m = matrix.shape[1]
    next_power_of_two = lambda x: 2**int(np.ceil(np.log2(x)))
    
    size = next_power_of_two(max(n, m))
    padded = np.zeros((size, size), dtype=np.float32)
    padded[:n, :m] = matrix
    return padded, n, m

# Función para eliminar el padding de la matriz
def unpad_matrix(matrix, original_n, original_m):
    """Elimina el relleno de la matriz para volver a las dimensiones originales"""
    return matrix[:original_n, :original_m]

def verify_results(standard_result, strassen_result, tolerance=1e-4):
    """Verifica que los resultados de ambos métodos sean similares"""
    # Tomar una pequeña muestra para comparar (para matrices muy grandes)
    max_sample = min(100, standard_result.shape[0])
    sample_indices = np.random.choice(standard_result.shape[0], max_sample, replace=False)
    
    sample_standard = standard_result[sample_indices, :][:, sample_indices]
    sample_strassen = strassen_result[sample_indices, :][:, sample_indices]
    
    return np.allclose(sample_standard, sample_strassen, rtol=tolerance, atol=tolerance)

def main():
    # Tamaños de matrices grandes
    sizes = [2048, 4096, 8192]
    
    print("\nComparación de algoritmos de multiplicación de matrices grandes en GPU")
    print("=" * 85)
    print(f"{'Tamaño':^10} | {'GPU Estándar (s)':^15} | {'GPU Strassen Optimizado (s)':^25} | {'Aceleración':^15} | {'Verificado':^10}")
    print("-" * 85)
    
    for size in sizes:
        try:
            print(f"Probando matrices de tamaño {size}x{size}...")
            
            # Crear matrices aleatorias
            A = np.random.rand(size, size).astype(np.float32)
            B = np.random.rand(size, size).astype(np.float32)
            
            # Medir tiempo para multiplicación estándar en GPU
            standard_result, standard_time = matrix_mul_gpu_standard(A, B)
            print(f"  - Multiplicación estándar completada: {standard_time:.6f} segundos")
            
            # Limpiar memoria
            cuda.current_context().deallocations.clear()
            gc.collect()
            
            # Preparar matrices para Strassen (padding a potencia de 2)
            A_padded, original_n, original_m = pad_matrix(A)
            B_padded, _, _ = pad_matrix(B)
            
            # Medir tiempo para el algoritmo de Strassen optimizado
            cuda.synchronize()
            start_time = time.perf_counter()
            strassen_result_padded = strassen_gpu_optimized(A_padded, B_padded)
            strassen_result = unpad_matrix(strassen_result_padded, size, size)
            cuda.synchronize()
            strassen_time = time.perf_counter() - start_time
            print(f"  - Multiplicación Strassen completada: {strassen_time:.6f} segundos")
            
            # Limpiar memoria
            del A_padded, B_padded, strassen_result_padded
            cuda.current_context().deallocations.clear()
            gc.collect()
            
            # Verificar resultados y calcular aceleración
            verified = verify_results(standard_result, strassen_result)
            speedup = standard_time / strassen_time if strassen_time > 0 else float('inf')
            
            print(f"{size:^10} | {standard_time:^15.6f} | {strassen_time:^25.6f} | {speedup:^15.3f} | {verified:^10}")
            
            # Liberar memoria
            del A, B, standard_result, strassen_result
            gc.collect()
            
        except cuda.CudaAPIError as e:
            print(f"Error de CUDA para tamaño {size}: {e}")
        except MemoryError:
            print(f"Error de memoria para tamaño {size}. Saltando al siguiente tamaño.")
        
        # Forzar limpieza de memoria
        cuda.current_context().deallocations.clear()
        gc.collect()

if __name__ == "__main__":
    main() 