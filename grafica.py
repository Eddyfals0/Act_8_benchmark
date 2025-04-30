import matplotlib.pyplot as plt

# Resultados de 10 ejecuciones (puedes modificarlos según tus pruebas)
cpu_times = [9.2986, 9.3417, 9.4342, 9.2457, 9.4802, 9.2146, 9.4041, 9.3407, 9.7667, 9.4475]
gpu_times = [0.2499, 0.2499, 0.2340, 0.2555, 0.2463, 0.2346, 0.2360, 0.2678, 0.2500, 0.2551]

# Calcular promedios
avg_cpu = sum(cpu_times) / len(cpu_times)
avg_gpu = sum(gpu_times) / len(gpu_times)

# Gráfica de barras de los promedios
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(['CPU (Promedio)', 'GPU (Promedio)'], [avg_cpu, avg_gpu], color=['red', 'blue'])
plt.ylabel('Tiempo (segundos)')
plt.title('Tiempo promedio de CPU vs GPU')
plt.grid(axis='y')
plt.text(0, avg_cpu + 0.1, f"{avg_cpu:.2f}s", ha='center', fontweight='bold')
plt.text(1, avg_gpu + 0.1, f"{avg_gpu:.2f}s", ha='center', fontweight='bold')

# Gráfica de línea de los 10 resultados individuales
plt.subplot(1, 2, 2)
plt.plot(cpu_times, label='CPU', marker='o', color='red')
plt.plot(gpu_times, label='GPU', marker='o', color='blue')
plt.xlabel('Iteración')
plt.ylabel('Tiempo (segundos)')
plt.title('Comparación por iteración')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
