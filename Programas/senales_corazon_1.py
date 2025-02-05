import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, welch
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
 
plt.close("all")
 
# Cargar los datos desde el archivo
file_path = r"D:\Upiita\6to\Patrones\Objetos\Signals10seg.txt"
data = np.loadtxt(file_path)
 
# Parámetros
fs = 2000  # Frecuencia de muestreo
t = np.arange(data.shape[0]) / fs  # Vector de tiempo basado en la frecuencia de muestreo
 
# Visualizar las señales
plt.figure(figsize=(15, 8))
for i in range(data.shape[1]):
    plt.subplot(6, 1, i + 1)
    plt.plot(t, data[:, i])
    plt.title(f'Señal {i + 1}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid()
 
plt.tight_layout()
# plt.show()
 
# Análisis en el dominio de la frecuencia
plt.figure(figsize=(15, 10))
frequencies = np.fft.rfftfreq(data.shape[0], d=1/fs)  # Frecuencias correspondientes
max_frequency = 600  # Frecuencia máxima a mostrar
for i in range(data.shape[1]):
    fft_magnitude = np.abs(np.fft.rfft(data[:, i]))  # Magnitud de la FFT
 
    # Filtrar hasta 600 Hz
    freq_mask = frequencies <= max_frequency
    filtered_frequencies = frequencies[freq_mask]
    filtered_magnitude = fft_magnitude[freq_mask]
 
    # Graficar
    plt.subplot(6, 1, i + 1)
    plt.plot(filtered_frequencies, filtered_magnitude)
    plt.title(f'Espectro de Frecuencia de la Señal {i + 1} (hasta 600 Hz)')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud')
    plt.grid()
 
plt.tight_layout()
plt.show()