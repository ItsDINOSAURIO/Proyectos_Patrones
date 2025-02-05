#Grabar audio
from scipy.io import wavfile 
import matplotlib.pyplot as plt
from winsound import *

import pyaudio
import wave
import numpy as np

#Parametros
formato=pyaudio.paInt16
canales=1
tasa_muestreo=int(44100/1)
tamano_bloque=1024
tiempo_grabacion=2
nombre_archivo='voz.wav'

audio=pyaudio.PyAudio()

# Listar dispositivos de entrada
print("Dispositivos disponibles:")
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    print(f"ID: {i}, Nombre: {info['name']}")

# Seleccionar el dispositivo de entrada
index_selected = int(input("Introduce el ID del micrófono a utilizar: "))

try:
    flujo_sonido=audio.open(format=formato,channels=canales,rate=tasa_muestreo,
                            input=True,frames_per_buffer=tamano_bloque,input_device_index=index_selected)
    print('Inicia la grabación')
    datos_audio=[]
    fragmentos=[]
    for i in range(0,int(tasa_muestreo/tamano_bloque*tiempo_grabacion)):
        datos_bloque=flujo_sonido.read(tamano_bloque)
        datos_audio.append(datos_bloque)
        fragmentos.append(np.frombuffer(datos_bloque,dtype = np.int16))
    senal_audio = np.hstack(fragmentos)
    if np.max(np.abs(senal_audio))!=0:
        senal_audio=senal_audio/np.max(np.abs(senal_audio))*32767 #El 32767 proviene de la magnitud debida al tipo de dato "int16" 2^16 y se trabaja con 16 bits para una mejor resolución (Se le sube el volumen)
        senal_audio=senal_audio.astype(np.int16)

    print('Termina la Grabación')
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

senal_normal=senal_audio/np.max(np.abs(senal_audio))  #normalizada
Bin1=np.where(np.abs(senal_normal)>=0.1,1,0)
ventana=440 #Tamaño de la ventana es de 10 mseg 
senal1=np.convolve(Bin1,np.ones(ventana)/ventana,mode='same')
Bin2=np.where(np.abs(senal1)>=0.1,1,0)
senal_activa=senal_normal[Bin2==1]

plt.figure(1)
plt.plot(senal_activa)

#Filtro de Pre énfasis
alpha=0.95 #Factor de pre énfasis
pre=np.roll(senal_activa,1)-alpha*senal_activa
pre[0]=0 #Primer valor en cero tras el desplazamiento
pre=np.where(np.abs(pre)>=0.7,0,pre) #Elimina cierta varianza aguda que se puede presentar por el tipo de micrófono a utilizar
plt.figure(2)
plt.plot(pre)
plt.title('Forma de Onda de la Grabación')

frame=441
overlap=397
ventanas=range(0,len(pre)-frame+1,overlap)
espectro=[]

for i in ventanas:
    segmento=pre[i:i+frame]*np.hamming(frame)
    Fourier=np.abs(np.fft.fft(segmento))[:150]
    espectro.append(Fourier)

Z=np.array(espectro).T #Transpuesta para cada columna sea el tiempo
plt.figure(3)
plt.imshow(Z,aspect='auto',origin='lower',cmap='viridis',extent=[0,len(ventanas),0,150])
plt.colorbar(label='Amplitud')
plt.title('Espectrograma de Fourier')

#Vista en 3 dimensiones
X,Y=np.meshgrid(np.arange(Z.shape[1]),np.arange(Z.shape[0]))
fig=plt.figure(4)
ax=fig.add_subplot(111,projection='3d')
surf=ax.plot_surface(X,Y,Z,cmap='viridis',edgecolor='none')
ax.set_xlabel('tiempo')

#Generar el patrón de las palabras

comp1=np.max(np.array(espectro),axis=0)
comp2=np.mean(np.array(espectro),axis=0)
patron=(comp1+comp2)/2
plt.figure(5)
plt.plot(patron)
plt.title('vector Patron')

muestra=1
#Grabar base de datos
# nombre_archivo=f"Jugo_0{muestra}.npy"#f"Cafe_0{muestra}.npy"#f"Agua_0{muestra}.npy"
# np.save(nombre_archivo,patron)
#Cargar base de datos
# palabra1=f"Agua_0{muestra}.npy"
# palabra2=f"Cafe_0{muestra}.npy"
# palabra3=f"Jugo_0{muestra}.npy"
# p1=np.load(palabra1)
# p2=np.load(palabra2)
# p3=np.load(palabra3)

# print(np.corrcoef(patron,p1))
# print(np.corrcoef(patron,p2))
# print(np.corrcoef(patron,p3))

#Clasificador if argmax, entrenar arquitectura neuronal
# plt.show()