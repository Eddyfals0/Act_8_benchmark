import numpy as np
import time
import os
import multiprocessing
from functools import partial

def multiply_matrices_sequential(A, B):
    """Multiplicación de matrices tradicional secuencial"""
    n = A.shape[0]
    C = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

def split_matrix(matrix):
    """Divide una matriz en cuatro submatrices"""
    n = matrix.shape[0] // 2
    return matrix[:n, :n], matrix[:n, n:], matrix[n:, :n], matrix[n:, n:]

def combine_matrices(C11, C12, C21, C22):
    """Combina cuatro submatrices en una matriz grande"""
    n = C11.shape[0]
    C = np.zeros((2*n, 2*n))
    C[:n, :n] = C11
    C[:n, n:] = C12
    C[n:, :n] = C21
    C[n:, n:] = C22
    return C

def strassen_sequential(A, B):
    """Algoritmo de Strassen secuencial"""
    n = A.shape[0]
    
    # Caso base: matrices 1x1
    if n == 1:
        return A * B
    
    # Dividir matrices en cuadrantes
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)
    
    # Calcular productos intermedios P1 a P7
    P1 = strassen_sequential(A11 + A22, B11 + B22)
    P2 = strassen_sequential(A21 + A22, B11)
    P3 = strassen_sequential(A11, B12 - B22)
    P4 = strassen_sequential(A22, B21 - B11)
    P5 = strassen_sequential(A11 + A12, B22)
    P6 = strassen_sequential(A21 - A11, B11 + B12)
    P7 = strassen_sequential(A12 - A22, B21 + B22)
    
    # Calcular cuadrantes del resultado
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    
    # Combinar cuadrantes
    return combine_matrices(C11, C12, C21, C22)

def process_subproblem(args):
    """Función auxiliar para paralelizar Strassen"""
    return strassen_sequential(*args)

def strassen_parallel(A, B, min_size=64):
    """Versión paralela del algoritmo de Strassen utilizando OpenMP/multiprocessing"""
    n = A.shape[0]
    
    # Usar algoritmo secuencial para matrices pequeñas
    if n <= min_size:
        return strassen_sequential(A, B)
    
    # Dividir matrices en cuadrantes
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)
    
    # Preparar los argumentos para los productos intermedios
    args = [
        (A11 + A22, B11 + B22),  # P1
        (A21 + A22, B11),        # P2
        (A11, B12 - B22),        # P3
        (A22, B21 - B11),        # P4
        (A11 + A12, B22),        # P5
        (A21 - A11, B11 + B12),  # P6
        (A12 - A22, B21 + B22)   # P7
    ]
    
    # Paralelizar el cálculo de P1-P7 usando multiprocessing
    with multiprocessing.Pool() as pool:
        results = pool.map(process_subproblem, args)
    
    P1, P2, P3, P4, P5, P6, P7 = results
    
    # Calcular cuadrantes del resultado
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    
    # Combinar cuadrantes
    return combine_matrices(C11, C12, C21, C22)

def pad_matrix(matrix):
    """Rellena la matriz para que tenga dimensiones potencia de 2"""
    n = matrix.shape[0]
    m = matrix.shape[1]
    next_power_of_two = lambda x: 2**int(np.ceil(np.log2(x)))
    
    # Encontrar la siguiente potencia de 2 mayor o igual a max(n, m)
    size = next_power_of_two(max(n, m))
    
    padded = np.zeros((size, size))
    padded[:n, :m] = matrix
    return padded, n, m

def unpad_matrix(matrix, original_n, original_m):
    """Elimina el relleno de la matriz para volver a las dimensiones originales"""
    return matrix[:original_n, :original_m]

def main():
    # Configurar tamaños de matriz para pruebas
    sizes = [64, 128, 256, 512]
    
    print("Comparación de algoritmos de multiplicación de matrices en CPU")
    print("=" * 60)
    print(f"Número de núcleos disponibles: {multiprocessing.cpu_count()}")
    print("-" * 60)
    print(f"{'Tamaño':^10} | {'Secuencial (s)':^15} | {'Strassen (s)':^15} | {'Aceleración':^10}")
    print("-" * 60)
    
    for size in sizes:
        # Crear matrices aleatorias
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)
        
        # Verificación con numpy (solo para comprobar)
        numpy_result = np.matmul(A, B)
        
        # Algoritmo secuencial
        start_time = time.time()
        sequential_result = multiply_matrices_sequential(A, B)
        sequential_time = time.time() - start_time
        
        # Verificar precisión del algoritmo secuencial
        assert np.allclose(sequential_result, numpy_result, rtol=1e-5), "Error en el algoritmo secuencial"
        
        # Preparar matrices para Strassen (necesita tamaño potencia de 2)
        A_padded, original_n, original_m = pad_matrix(A)
        B_padded, _, _ = pad_matrix(B)
        
        # Algoritmo de Strassen paralelo
        start_time = time.time()
        strassen_result_padded = strassen_parallel(A_padded, B_padded)
        strassen_result = unpad_matrix(strassen_result_padded, size, size)
        strassen_time = time.time() - start_time
        
        # Verificar precisión del algoritmo de Strassen
        if not np.allclose(strassen_result, numpy_result, rtol=1e-5):
            print(f"Error en Strassen para tamaño {size}")
        
        # Calcular aceleración
        speedup = sequential_time / strassen_time if strassen_time > 0 else float('inf')
        
        # Imprimir resultados
        print(f"{size:^10} | {sequential_time:^15.6f} | {strassen_time:^15.6f} | {speedup:^10.2f}")
    
    print("-" * 60)

if __name__ == "__main__":
    main()