import numpy as np
import time
from numba import cuda
import math
import warnings
from numba.cuda.cudadrv.devicearray import NumbaPerformanceWarning

# Suprimir advertencias de Numba
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

# Configuración para GPU
BLOCK_SIZE = 16  # Tamaño de bloque para kernels CUDA (optimizado para GTX 960M)

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
    
    return C, end_time - start_time

# Kernel CUDA para multiplicación de submatrices de Strassen
@cuda.jit
def strassen_kernel(A, B, C, threshold):
    """Kernel CUDA para multiplicación de submatrices usando Strassen"""
    row, col = cuda.grid(2)

    # Si estamos dentro de los límites de la matriz
    if row < C.shape[0] and col < C.shape[1]:
        # Realizar las multiplicaciones de submatrices
        if A.shape[0] <= threshold:
            tmp = 0.0
            for k in range(A.shape[1]):
                tmp += A[row, k] * B[k, col]
            C[row, col] = tmp
        else:
            # Aquí iría el código para manejar las submatrices, pero puede ser reemplazado
            # por un conjunto de kernels que manejen cada parte del Strassen.
            pass

# Función para realizar Strassen en GPU
def strassen_gpu(A, B, threshold=64):
    """Algoritmo de Strassen en GPU con umbral para cambiar a multiplicación estándar"""
    n = A.shape[0]

    # Si la matriz es lo suficientemente pequeña, usar el método estándar en GPU
    if n <= threshold:
        result, _ = matrix_mul_gpu_standard(A, B)
        return result
    
    # Dividir matrices en cuadrantes
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    # Calcular productos intermedios P1 a P7 en paralelo usando CUDA
    P1 = strassen_gpu(A11 + A22, B11 + B22, threshold)
    P2 = strassen_gpu(A21 + A22, B11, threshold)
    P3 = strassen_gpu(A11, B12 - B22, threshold)
    P4 = strassen_gpu(A22, B21 - B11, threshold)
    P5 = strassen_gpu(A11 + A12, B22, threshold)
    P6 = strassen_gpu(A21 - A11, B11 + B12, threshold)
    P7 = strassen_gpu(A12 - A22, B21 + B22, threshold)

    # Calcular cuadrantes del resultado
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    
    # Combinar cuadrantes y devolver el resultado
    return combine_matrices(C11, C12, C21, C22)

# Función para dividir una matriz en cuatro submatrices
def split_matrix(matrix):
    """Divide una matriz en cuatro submatrices"""
    n = matrix.shape[0] // 2
    return matrix[:n, :n], matrix[:n, n:], matrix[n:, :n], matrix[n:, n:]

# Función para combinar cuatro submatrices en una matriz grande
def combine_matrices(C11, C12, C21, C22):
    """Combina cuatro submatrices en una matriz grande"""
    n = C11.shape[0]
    C = np.zeros((2*n, 2*n), dtype=np.float32)
    C[:n, :n] = C11
    C[:n, n:] = C12
    C[n:, :n] = C21
    C[n:, n:] = C22
    return C

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

# Función para dividir una matriz en cuatro submatrices
def split_matrix(matrix):
    """Divide una matriz en cuatro submatrices"""
    n = matrix.shape[0] // 2
    return matrix[:n, :n], matrix[:n, n:], matrix[n:, :n], matrix[n:, n:]

# Función para combinar cuatro submatrices en una matriz grande
def combine_matrices(C11, C12, C21, C22):
    """Combina cuatro submatrices en una matriz grande"""
    n = C11.shape[0]
    C = np.zeros((2*n, 2*n), dtype=np.float32)
    C[:n, :n] = C11
    C[:n, n:] = C12
    C[n:, :n] = C21
    C[n:, n:] = C22
    return C

# Función para multiplicar matrices en GPU
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
    
    return C, end_time - start_time

# Función para realizar Strassen en GPU
def strassen_gpu(A, B, threshold=64):
    """Algoritmo de Strassen en GPU con umbral para cambiar a multiplicación estándar"""
    A = np.ascontiguousarray(A)  # Asegurar que A sea contigua
    B = np.ascontiguousarray(B)  # Asegurar que B sea contigua
    
    n = A.shape[0]

    # Si la matriz es lo suficientemente pequeña, usar el método estándar en GPU
    if n <= threshold:
        result, _ = matrix_mul_gpu_standard(A, B)
        return result
    
    # Dividir matrices en cuadrantes
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    # Calcular productos intermedios P1 a P7 en paralelo usando CUDA
    P1 = strassen_gpu(A11 + A22, B11 + B22, threshold)
    P2 = strassen_gpu(A21 + A22, B11, threshold)
    P3 = strassen_gpu(A11, B12 - B22, threshold)
    P4 = strassen_gpu(A22, B21 - B11, threshold)
    P5 = strassen_gpu(A11 + A12, B22, threshold)
    P6 = strassen_gpu(A21 - A11, B11 + B12, threshold)
    P7 = strassen_gpu(A12 - A22, B21 + B22, threshold)

    # Calcular cuadrantes del resultado
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    
    # Combinar cuadrantes y devolver el resultado
    return combine_matrices(C11, C12, C21, C22)

# Función para eliminar el padding de la matriz
def unpad_matrix(matrix, original_n, original_m):
    """Elimina el relleno de la matriz para volver a las dimensiones originales"""
    return matrix[:original_n, :original_m]


# Función principal para ejecutar la comparación
def main():
    sizes = [64, 128, 256, 512, 1024, 2048]
    
    print("\nComparación de algoritmos de multiplicación de matrices en GPU")
    print("=" * 60)
    print(f"{'Tamaño':^10} | {'GPU Estándar (s)':^15} | {'GPU Strassen (s)':^15} | {'Aceleración':^10}")
    print("-" * 60)
    
    for size in sizes:
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)
        
        numpy_result = np.matmul(A, B)
        
        _, standard_time = matrix_mul_gpu_standard(A, B)
        
        A_padded, original_n, original_m = pad_matrix(A)
        B_padded, _, _ = pad_matrix(B)
        
        cuda.synchronize()
        start_time = time.perf_counter()
        strassen_result_padded = strassen_gpu(A_padded, B_padded)
        strassen_result = unpad_matrix(strassen_result_padded, size, size)
        cuda.synchronize()
        strassen_time = time.perf_counter() - start_time
        
        strassen_accurate = np.allclose(strassen_result, numpy_result, rtol=1e-4, atol=1e-4)
        
        speedup = round(standard_time / strassen_time, 3) if strassen_time > 0 else 'N/A'
        
        if speedup == 'N/A':
            print(f"{size:^10} | {standard_time:^15.6f} | {strassen_time:^15.6f} | {speedup:^10}")
        else:
            print(f"{size:^10} | {standard_time:^15.6f} | {strassen_time:^15.6f} | {speedup:^10.3f}")

if __name__ == "__main__":
    main()
