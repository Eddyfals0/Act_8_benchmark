# Importación de bibliotecas necesarias
import multiprocessing  # Para manejo de múltiples procesos
import numpy as np      # Para operaciones numéricas eficientes
import time            # Para medir el tiempo de ejecución

# Tamaño de los vectores a sumar
size = 100_000

def add_chunk(start, end, a, b, result):
    """
    Suma una porción (chunk) de dos vectores y almacena el resultado.
    
    Args:
        start: Índice inicial del chunk
        end: Índice final del chunk
        a, b: Vectores a sumar
        result: Array donde se almacenará el resultado
    """
    result[start:end] = a[start:end] + b[start:end]

def benchmark_cpu():
    """
    Realiza una prueba de rendimiento de la suma de vectores usando múltiples procesos.
    Crea dos vectores aleatorios, los suma en paralelo y mide el tiempo de ejecución.
    """
    # Creación de vectores aleatorios
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    
    # Array compartido para almacenar resultados
    result = multiprocessing.Array('f', size)

    # Determinación del número de procesos según los cores disponibles
    num_processes = multiprocessing.cpu_count()
    chunk_size = size // num_processes
    processes = []

    # Inicio de la medición de tiempo
    start_time = time.time()
    
    # Creación y lanzamiento de procesos
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = size if i == num_processes - 1 else (i + 1) * chunk_size
        p = multiprocessing.Process(target=add_chunk, args=(start_idx, end_idx, a, b, result))
        processes.append(p)
        p.start()

    # Espera a que todos los procesos terminen
    for p in processes:
        p.join()
    
    # Fin de la medición de tiempo
    end_time = time.time()

    # Impresión del tiempo de ejecución
    print(f"[CPU] Tiempo de ejecución con {num_processes} cores: {end_time - start_time:.4f} segundos")

if __name__ == "__main__":
    benchmark_cpu()
