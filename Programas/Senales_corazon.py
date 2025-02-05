#Partir las 6 a la mitad en 5 y 5 segundos, para tener 12 señales de 5 segundos plicar DTW para hacer todas contra todas y clasificar entre sanas y no sans
# import numpy as np
# import matplotlib.pyplot as plt

# señales=r"D:\Upiita\6to\Patrones\Objetos\Signals10seg.txt"
# señales=np.loadtxt(señales)
# # print(len(señales))
# # print(np.size(señales))

# plt.plot(señales[:,1])
# plt.show()

# FS=2000
# import numpy as np
# import matplotlib.pyplot as plt

# # Cargar las señales
# ruta_señales = r"D:\Upiita\6to\Patrones\Objetos\Signals10seg.txt"
# señales = np.loadtxt(ruta_señales)

# # Definir parámetros
# num_señales = 6  # Número total de señales de 10 segundos
# segmentos = 2    # Número de segmentos por señal (10 segundos -> 2 segmentos de 5 segundos)
# muestras_por_segmento = len(señales) // (num_señales * segmentos)

# # Crear los 12 segmentos
# segmentos_5seg = []
# for i in range(num_señales):
#     inicio_1 = i * 2 * muestras_por_segmento
#     inicio_2 = inicio_1 + muestras_por_segmento
#     segmentos_5seg.append(señales[inicio_1:inicio_1 + muestras_por_segmento, 1])  # Primer segmento
#     segmentos_5seg.append(señales[inicio_2:inicio_2 + muestras_por_segmento, 1])  # Segundo segmento

# # Graficar los 12 segmentos en subplots
# fig, axes = plt.subplots(6, 2, figsize=(10, 15))
# fig.suptitle("Segmentos de 5 segundos", fontsize=16)

# # Añadir cada segmento al subplot
# for i, segmento in enumerate(segmentos_5seg):
#     fila = i // 2
#     columna = i % 2
#     axes[fila, columna].plot(segmento)
#     axes[fila, columna].set_title(f"Segmento {i + 1}")
#     axes[fila, columna].grid()

# # Ajustar espacio
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Cargar las señales
ruta_señales = r"D:\Upiita\6to\Patrones\Objetos\Signals10seg.txt"
señales = np.loadtxt(ruta_señales)

# Parámetros
fs = 2000  # Frecuencia de muestreo en Hz (ejemplo: 100 muestras por segundo)
duracion_segmento = 5  # Duración de cada segmento en segundos
muestras_por_segmento = fs * duracion_segmento

num_señales = señales.shape[1]  # Número de señales (columnas)
segmentos_por_señal = 2  # Cada señal de 10 segundos se divide en 2 segmentos de 5 segundos
total_segmentos = num_señales * segmentos_por_señal

# Crear los segmentos
segmentos_5seg = []
for i in range(num_señales):
    for j in range(segmentos_por_señal):
        inicio = j * muestras_por_segmento + i * muestras_por_segmento * segmentos_por_señal
        fin = inicio + muestras_por_segmento
        segmentos_5seg.append(señales[inicio:fin, i])

print(segmentos_5seg)
# Graficar los segmentos en subplots
fig, axes = plt.subplots(6, 2, figsize=(10, 15))
fig.suptitle("Segmentos de 5 segundos (Basados en Frecuencia de Muestreo)", fontsize=16)

# Añadir cada segmento al subplot
for idx, segmento in enumerate(segmentos_5seg):
    print(segmento)
    fila = idx // 2
    columna = idx % 2
    axes[fila, columna].plot(np.linspace(0, duracion_segmento, muestras_por_segmento), segmento)
    axes[fila, columna].set_title(f"Segmento {idx + 1}")
    axes[fila, columna].set_xlabel("Tiempo (s)")
    axes[fila, columna].set_ylabel("Amplitud")
    axes[fila, columna].grid()

# Ajustar espacio
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
