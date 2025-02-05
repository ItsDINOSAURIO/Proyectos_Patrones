import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from scipy.signal import butter, lfilter, spectrogram, find_peaks

#Filtro Pasabajas
def butter_filter(data, cutoff, fs, order=5, btype='low'):
    nyquist = 0.5 * fs
    if isinstance(cutoff, (list, tuple, np.ndarray)):
        # Normalizar las frecuencias de corte para filtros de banda
        normal_cutoff = [c / nyquist for c in cutoff]
    else:
        # Normalizar la frecuencia de corte para filtros de un solo polo
        normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    y = lfilter(b, a, data)
    return y

# Función para introducir un delay en la señal
def introduce_delay(data, delay_samples):
    delayed_signal = np.roll(data, delay_samples)
    # Para los valores retrasados que salen del array, los ponemos en cero
    delayed_signal[:delay_samples] = 0
    return delayed_signal

ruta_señal = r"D:\Upiita\6to\Patrones\Objetos\audiocorazon1.wav"

# Cargar la señal de sonido
fs, señal = wavfile.read(ruta_señal)  # fs es la frecuencia de muestreo
# señal=señal[46167444:]
 
# Verificar información de la señal
print(f"Frecuencia de muestreo: {fs} Hz")
print(f"Duración de la señal: {len(señal) / fs} segundos")

if len(señal.shape) == 2:
    señal = señal.mean(axis=1)
 
# Normalizar la señal
señal_normalizada = señal / np.max(np.abs(señal))
 
# Verificar la normalización
print(f"Valor máximo después de normalización: {np.max(señal_normalizada)}")
 
# Crear un vector de tiempo
tiempo = np.linspace(0, len(señal) / fs, num=len(señal))
 
# Graficar la señal original y normalizada
plt.close('all')
plt.figure(figsize=(14, 6))
 
plt.subplot(2, 1, 1)
plt.plot(tiempo, señal)
plt.title('Señal Original')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
 
plt.subplot(2, 1, 2)
plt.plot(tiempo, señal_normalizada)
plt.title('Señal Normalizada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
 
plt.tight_layout()

# Escalado estándar de la señal normalizada
scaler = StandardScaler()
señal_escalada = scaler.fit_transform(señal_normalizada.reshape(-1, 1)).flatten()
 
# Verificar el escalado
print(f"Media de la señal escalada: {np.mean(señal_escalada)}")
print(f"Desviación estándar de la señal escalada: {np.std(señal_escalada)}")
 
# Graficar la señal escalada
plt.figure(figsize=(10, 4))
plt.plot(tiempo, señal_escalada)
plt.title('Señal Escalada para ICA')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud Escalada')
plt.tight_layout()

# a) Introducir diferentes retrasos
delay_samples = int(0.001 * fs)  # 1 ms de retraso
señal_delay1 = introduce_delay(señal_escalada, delay_samples)
señal_delay2 = introduce_delay(señal_escalada, 2 * delay_samples)
 
# b) Aplicar diferentes filtros
señal_low = butter_filter(señal_escalada, cutoff=500, fs=fs, btype='low')  # Pasa-bajo 500 Hz
señal_high = butter_filter(señal_escalada, cutoff=1500, fs=fs, btype='high')  # Pasa-alto 1500 Hz
4
# c) Añadir ruido blanco de baja amplitud
np.random.seed(0)  # Para reproducibilidad
ruido = np.random.normal(0, 0.01, len(señal_escalada))
señal_noise = señal_escalada + ruido

# Organizar las versiones en una matriz donde cada columna es una versión diferente
mezcla = np.vstack((señal_escalada, señal_delay1, señal_delay2, señal_low, señal_high, señal_noise)).T
 
print(f"Dimensiones de la matriz de mezcla: {mezcla.shape}")

n_componentes = 3  # Por ejemplo, 2 componentes independientes, simula que la señal está compuesta de n señales
 
# Inicializar FastICA
ica = FastICA(n_components=n_componentes, random_state=0)
 
# Ajustar el modelo ICA
componentes = ica.fit_transform(mezcla)  # Señales independientes, aplicar finspeaks para determinar la frecuencia cardiaca
mezcla_inversa = ica.inverse_transform(componentes)
 
print("Componentes independientes extraídas mediante ICA.")
 
# 4. Visualización de las Componentes Extraídas
 
# Crear un nuevo vector de tiempo para las componentes
tiempo_componentes = tiempo[:len(componentes)]

tiempo_componentes = tiempo[:len(componentes)]
 
plt.figure(figsize=(14, 8))
 
for i in range(componentes.shape[1]):
    plt.subplot(componentes.shape[1], 1, i+1)
    plt.plot(tiempo_componentes, componentes[:, i])
    plt.title(f'Componente Independiente {i+1}')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
 
plt.tight_layout()

# Por ejemplo, graficar el espectrograma de cada componente
plt.figure(figsize=(14, 8))
 
for i in range(componentes.shape[1]):
    f, t_spec, Sxx = spectrogram(componentes[:, i], fs)
    plt.subplot(componentes.shape[1], 1, i+1)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')  # Añadido 1e-10 para evitar log(0)
    plt.title(f'Spectrograma de la Componente Independiente {i+1}')
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    plt.colorbar(label='Intensidad [dB]')
 
plt.tight_layout()

indice_componente_corazon = 1  # 0 para la primera componente, 1 para la segunda, etc.
 
componente_corazon = componentes[:, indice_componente_corazon]
 
# Definir un rango de frecuencia para el filtro de banda (ejemplo: 300 Hz a 1500 Hz)
frecuencia_inferior = 300
frecuencia_superior = 1500
 
# Aplicar filtro de banda correctamente
componente_corazon_filtrada = butter_filter(
    componente_corazon,
    cutoff=[frecuencia_inferior, frecuencia_superior],
    fs=fs,
    btype='band',
    order=4
)
 
# Graficar la componente filtrada
plt.figure(figsize=(10, 4))
plt.plot(tiempo_componentes, componente_corazon_filtrada)
plt.title('Componente Independiente Filtrada - Sonidos del Corazón')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.tight_layout()

señal_corazon = scaler.inverse_transform(componente_corazon_filtrada.reshape(-1, 1)).flatten()
 
# Normalizar nuevamente para evitar clipping
señal_corazon_normalizada = señal_corazon / np.max(np.abs(señal_corazon))
 
# Guardar la señal como un archivo WAV
ruta_guardado = 'sonidos_corazon.wav'
wavfile.write(ruta_guardado, fs, señal_corazon_normalizada.astype(np.float32))
 
print(f"Componente filtrada guardada como '{ruta_guardado}'.")

# Detectar picos en la señal filtrada (latidos)
distancia_minima = int(0.3 * fs)  # Asegurar un intervalo mínimo de 0.3 s (200 BPM máx)
altura_minima = np.max(componente_corazon_filtrada) * 0.5  # Filtrar solo los picos significativos

# Detectar picos que representan los latidos
picos, _ = find_peaks(componente_corazon_filtrada, distance=distancia_minima)#, height=altura_minima)

# Calcular la frecuencia cardíaca
intervalos = np.diff(picos) / fs  # Tiempo entre picos en segundos
frecuencia_cardiaca = 60 / intervalos  # Convertir a latidos por minuto (BPM)

# Verificar el resultado de los latidos
print(f"Picos detectados: {len(picos)}")
print(f"Frecuencia cardíaca media: {np.mean(frecuencia_cardiaca):.2f} BPM")

# Graficar la señal y los picos
plt.figure(figsize=(12, 4))
plt.plot(tiempo_componentes, componente_corazon_filtrada, label='Componente del Corazón Filtrada')
plt.plot(tiempo_componentes[picos], componente_corazon_filtrada[picos], 'rx', label='Picos (Latidos)')
plt.title('Detección de Latidos en la Señal de Sonido del Corazón')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.tight_layout()

# Graficar la frecuencia cardíaca estimada en función del tiempo
tiempos_inter_picos = tiempo_componentes[picos][1:]  # Ignorar el primer punto para diferenciar los intervalos
plt.figure(figsize=(10, 4))
plt.plot(tiempos_inter_picos, frecuencia_cardiaca, label='Frecuencia Cardíaca (BPM)', marker='o')
plt.title('Frecuencia Cardíaca Estimada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia Cardíaca [BPM]')
plt.legend()
plt.tight_layout()

plt.show()