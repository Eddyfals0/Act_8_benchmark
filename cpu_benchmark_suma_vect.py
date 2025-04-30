# Importación de bibliotecas necesarias
import multiprocessing  # Para manejo de múltiples procesos
import numpy as np      # Para operaciones numéricas eficientes
import time            # Para medir el tiempo de ejecución
import ctypes          # Para tipos de datos compatibles con C

# Tamaño de los vectores a sumar
size = 100_000_000

def add_chunk(start, end, shared_a, shared_b, shared_result):
    """
    Suma una porción (chunk) de dos vectores y almacena el resultado.
    
    Args:
        start: Índice inicial del chunk
        end: Índice final del chunk
        shared_a: Vector A compartido
        shared_b: Vector B compartido
        shared_result: Vector resultado compartido
    """
    # Realizamos la suma
    for i in range(start, end):
        shared_result[i] = shared_a[i] + shared_b[i]

def benchmark_cpu():
    """
    Realiza una prueba de rendimiento de la suma de vectores usando múltiples procesos.
    Crea dos vectores aleatorios, los suma en paralelo y mide el tiempo de ejecución.
    """
    # Creación de vectores aleatorios
    a_np = np.random.rand(size).astype(np.float32)
    b_np = np.random.rand(size).astype(np.float32)
    
    # Creamos arrays compartidos
    shared_a = multiprocessing.RawArray(ctypes.c_float, size)
    shared_b = multiprocessing.RawArray(ctypes.c_float, size)
    shared_result = multiprocessing.RawArray(ctypes.c_float, size)
    
    # Copiamos los datos a los arrays compartidos
    a_temp = np.frombuffer(shared_a, dtype=np.float32)
    b_temp = np.frombuffer(shared_b, dtype=np.float32)
    a_temp[:] = a_np[:]
    b_temp[:] = b_np[:]

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
        p = multiprocessing.Process(
            target=add_chunk, 
            args=(start_idx, end_idx, shared_a, shared_b, shared_result)
        )
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
    # Necesario para Windows
    multiprocessing.freeze_support()
    benchmark_cpu()
