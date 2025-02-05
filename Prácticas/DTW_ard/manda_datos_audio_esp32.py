# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 23:06:39 2024

@author: Robles
"""

from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from winsound import PlaySound, SND_FILENAME, SND_ASYNC
# from scipy.signal import find_peaks
import pyaudio
import time
import numpy as np
import wave

plt.close("all")
#Parametros de grabacion
formato = pyaudio.paInt16 #un valor, 16 bits para una mejor resolucion
canales = 1 #2 para estereo
tasa_muestreo = 4410 #valores en 1 seg, mp3 de calidad, biologica 4000 u 8000
tamano_bloque = 1024 #pedazos de 1024 datos
tiempo_grabacion = 2 #ojo 24 frames por segundo
nombre_archivo = ('prueba_voz.wav')


#tasa_muestreo, senal_audio = wav.read(nombre_archivo)


audio = pyaudio.PyAudio()
# si esta desocupada la tarjeta de audio (es valido), hace esto
try:
    flujo_sonido = audio.open(format=formato, channels=canales, rate=tasa_muestreo, input=True, frames_per_buffer=tamano_bloque)
    print('Inicia la grabación')
    datos_audio = []
    fragmentos = []
    for i in range(0, int(tasa_muestreo/tamano_bloque*tiempo_grabacion)):
        datos_bloque = flujo_sonido.read(tamano_bloque)
        datos_audio.append(datos_bloque)
        fragmentos.append(np.frombuffer(datos_bloque, dtype=np.int16)) 
    senal_audio = np.hstack(fragmentos) #apilar de forma horizontal para ver la señal
    if np.max(np.abs(senal_audio)) != 0: #si hay algo grabado
        senal_audio = senal_audio/np.max(np.abs(senal_audio))*32767 #espera valores de ese rango, sino la senal no es audible, le sube el volumen con el *32767
        senal_audio = senal_audio.astype(np.int16)
    print('Termina la Grabación')
    flujo_sonido.stop_stream()
    flujo_sonido.close()
finally:
    audio.terminate()
    
#almacenar y reproducir la grabacion
archivo_wave = wave.open(nombre_archivo, 'wb') #escribir en el nombre del archivo
archivo_wave.setnchannels(canales)
archivo_wave.setsampwidth(audio.get_sample_size(formato))
archivo_wave.setframerate(tasa_muestreo)
archivo_wave.writeframes(b''.join([senal_audio.tobytes()])) #unir por espacios bites 01101 01010
archivo_wave.close()

PlaySound(nombre_archivo, SND_FILENAME | SND_ASYNC)


#===== Quitarle puntos muertos =====
plt.figure()
plt.plot(senal_audio)
plt.title('Forma de Onda de la Señal original')
senal_normal = senal_audio/np.max(np.abs(senal_audio)) # senal normalizada de -1 a 1
bin1 = np.where(np.abs(senal_normal) >= 0.2, 1, 0) # lo mayor a 0.1 lo pones en 1, lo demas a 0 para quitar el espacio muerto
ventana = 440 # tamano de la ventana = 10 miliseg
senal1 = np.convolve(bin1, np.ones(ventana)/ventana, mode='same') # guarda los espacios con un cambio
bin2 = np.where(senal1 >= 0.2, 1, 0)
senal_activa = senal_normal[bin2 == 1]
plt.figure()
plt.plot(senal_activa)
plt.title('Forma de Onda de la Señal sin puntos muertos')

#===== Filtro de pre-enfaasis =====
# Se accentuqn o mejoran las vocales al desplazar a la señal original 1 espacio
alpha = 0.95 # factor de pre-enfasis
pre = np.roll (senal_activa, 1) - alpha * senal_activa
pre[0] = 0 # primer valor en cero tras el desplazamiento
pre = 100*np.where (np.abs(pre) >= 0.7, 0, pre)
plt.figure()
plt.plot(pre)
plt.title('Forma de Onda de la Señal con la vocal enfatizada')
plt.show()


# Enviar mensaje
import socket

ESP32_IP = "192.168.1.82"
ESP32_PORT = 80

def enviar_datos(mensaje):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ESP32_IP, ESP32_PORT))
        s.sendall((mensaje + '\n').encode())
        respuesta = s.recv(1024)
        print(f"Respuesta: {respuesta.decode()}")




# Dividir la señal en ventanas de 100 datos
ventana_tamano = 100
n_muestras = len(pre)
# Calcular cuántas ventanas caben en la señal
n_ventanas = int(np.ceil(n_muestras / ventana_tamano))
# Asegurarse de que la última ventana tenga suficientes muestras
for i in range(n_ventanas):
    start = i * ventana_tamano
    end = min((i + 1) * ventana_tamano, n_muestras)
    
    # Si es la última ventana y no tiene suficiente tamaño, rellena con ceros
    ventana = pre[start:end]
    if len(ventana) < ventana_tamano:
        ventana = np.pad(ventana, (0, ventana_tamano - len(ventana)), 'constant')
    
    result = ' '.join(map(lambda x: str(int(x)), ventana))  # Convertir cada elemento a int
    #print(result)
    enviar_datos(result)  # Llamada a la función para enviar los datos
    time.sleep(1)

enviar_datos("FIN")
print(f"Señal completa enviada: ({n_ventanas} ventanas)")
