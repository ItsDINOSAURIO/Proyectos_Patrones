#Reconocer quien dijo la palabra por mdeio del espectro de la señal almacenada, buscar que caracteristicas identifica un hombre a una mujer, pcd de fourier buscas
#Reconocedor de audio
#se puede manejar con 32 bits tambien 

from scipy.io import wavfile
from matplotlib import pyplot as plt
from winsound import *

import pyaudio 
import wave
import numpy as np
from scipy.signal import butter,filtfilt,correlate

def bandpass_filter(signal,lowcut,highcut,fs,order=6):
    nyq=0.5*320
    low=lowcut/nyq
    high=highcut/nyq
    b,a=butter(order,[low,high],btype='band')
    y=filtfilt(b,a,signal)
    return y

plt.close('all')

#Parametros de grabacion 
formato = pyaudio.paInt16
canales = 1
tasa_muestreo = 44100 #Señal mioelectrica 4000  y para audio de alta calidad 44100, mp3 
#pedazos de 1024 datos de la tasa de muestreo
tamano_bloque = 1024 # Señal de voz buscar la frecuencia 2 veces arriba de la frecuencia base el muestreo es mejor si se toma mayor a 2 veces la frecuencia base 
tiempo_grabacion = 3
nombre_archivo = 'voz.wav'
#

#Grabacion
audio = pyaudio.PyAudio() # Permite generar objetos de audio 
try: # excepciones del codigo para que no se trabe la maquina, para verificar si esta libre la computadora
    flujo_sonido = audio.open(format = formato, channels = canales, rate = tasa_muestreo, input = True, frames_per_buffer = tamano_bloque)# Abre el objeto, formato, numero de canales, la tasa de muestreo,inicie la entrada, tamaño de buffer
    print('Inicia la grabacion')
    datos_audio = []
    fragmentos = []
    for i in range(0,int(tasa_muestreo/tamano_bloque*tiempo_grabacion)):
        datos_bloque = flujo_sonido.read(tamano_bloque)#Tomando lo que hay en el buffer de la computadora 
        datos_audio.append(datos_bloque)  
        fragmentos.append(np.frombuffer(datos_bloque,dtype = np.int16))
    senal_audio = np.hstack(fragmentos)
    if np.max(np.abs(senal_audio))!=0:
        senal_audio = senal_audio/np.max(np.abs(senal_audio))*32767 #Esto sirve para aumentar la señal ya que esta solamente en un rango de 1 y -1, aumentas el volumen
        senal_audio = senal_audio.astype(np.int16)
    print('Termina la Grabacion')
    flujo_sonido.stop_stream()
    flujo_sonido.close()
finally:
    audio.terminate()

    
archivo_wave=wave.open(nombre_archivo,'wb')
archivo_wave.setnchannels(canales)
archivo_wave.setsampwidth(audio.get_sample_size(formato))
archivo_wave.setframerate(tasa_muestreo)
archivo_wave.writeframes(b''.join([senal_audio.tobytes()]))
archivo_wave.close()

PlaySound(nombre_archivo,SND_FILENAME|SND_ASYNC)

#Elimina los espacios muertos de la señal de audio que se guardo con anterioridad  
Senal_Normal = senal_audio/np.max(np.abs(senal_audio)) #Normaliza la señal 
Bin1 = np.where(np.abs(Senal_Normal) >= 0.1,1,0) # Calcula el absoluto lo que no esa mayor a 0.1 sea 0 y si no es asi que sea 1 
ventana = 440 # Tamaño de la ventana es de 10 miliseg 
Senal1 = np.convolve(Bin1,np.ones(ventana)/ventana,mode='same')
Bin2 = np.where(Senal1 >= 0.1,1,0)
senal_activa = Senal_Normal[Bin2 == 1]

plt.figure()
plt.plot(senal_activa)
plt.title('Forma de Onda de la Grabacion') 


#...... Filtro de Pre-enfasis enfatiza las altas frecuencias, mejora las frecuencias altas 
alpha = 0.95 # factor de pre-enfasis 
Pre = np.roll(senal_activa, 1)- alpha * senal_activa
Pre[0] = 0 #Primer valor en cero tras el desplazamiento 
Pre = np.where(np.abs(Pre) >= 0.7, 0, Pre) # Elimina los elementos que son muy agudas

Pre_fil = [0] * len(Pre) 

for i in range(len(Pre)):
    if Pre[i] > 0.01 or Pre[i] < -0.01:
        Pre_fil[i] = Pre[i]
    else:
        Pre_fil[i] = 0
        
lowcut=0.05
highcut=150.0
# Pre_fil=bandpass_filter(Pre_fil,lowcut,highcut,tasa_muestreo)
plt.figure()
plt.plot(Pre)
plt.title('Forma de Onda de la Grabacion pre procesada') 

plt.figure()
plt.plot(Pre_fil)
plt.title('Forma de Onda de la Grabacion filtrada') 

#---------------------------------------------------
frame = 441
overlap = 397 
ventanas = range(0,len(Pre) - frame + 1, overlap) # Agrega un translape 
espectro = []

for i in ventanas:
    segmento = Pre[i: i+frame] * np.hamming(frame)
    Fourier = np.abs(np.fft.fft(segmento))[:150]
    espectro.append(Fourier)
    
z = np.array(espectro).T # Transpuesta para que cada columna sea le tiempo     
plt.figure()
plt.imshow(z, aspect='auto', origin = 'lower', cmap='viridis', extent=[0, len(ventanas), 0, 150]) # 150 datos positivos de fourier 
plt.colorbar(label='Amplitud')
plt.title('Espectrograma de Fourier')
#-------- Vista en 3 dimensiones 
X, Y = np.meshgrid(np.arange(z.shape[1]),np.arange(z.shape[0]))
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
surf = ax.plot_surface(X,Y,z,cmap='viridis',edgecolor='none')
ax.set_xlabel('Tiempo')
ax.set_ylabel('Frecuencia')
ax.set_zlabel('Amplitud')

#Reconocimiento de patrones 
#Generar el patron de las palabras 

comp1 = np.max(np.array(espectro),axis = 0)
comp2 = np.mean(np.array(espectro),axis = 0)
patron = (comp1+comp2)/2
plt.figure()
plt.plot(patron)
plt.title('Vector patron')

fil_patron = []

for i in range(len(patron)):
    if patron[i] > 0.05:
        fil_patron.append(patron[i])   
        
plt.figure()
plt.plot(fil_patron)
plt.title('Vector patron filtrado')

#Sirve para guardar en un archivo np el vector 
# muestra = 10 
# nombre_archivo = f"Jugo_0{muestra}.npy" #Cambiar a cafe y juego 
# np.save(nombre_archivo,patron)


#Quitar los murmullos con la magnitud 
# plt.show()