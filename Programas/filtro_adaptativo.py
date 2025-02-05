from scipy.io import wavfile
from matplotlib import pyplot as plt
from winsound import PlaySound,SND_FILENAME,SND_ASYNC

import numpy as np
import time

muestreo1,voz=wavfile.read(r"D:\Upiita\6to\Patrones\Objetos\guitarra_voz\voz.wav")
t1=np.arange(len(voz))/float(muestreo1)
voz=voz/(2**15)
plt.close('all')
plt.figure(figsize=(18,4))
plt.plot(t1,voz,label='senal voz(persona)',color='b')
plt.legend()

# PlaySound(r"D:\Upiita\6to\Patrones\Objetos\guitarra_voz\voz.wav",SND_FILENAME|SND_ASYNC)

muestreo2,guitarra=wavfile.read(r"D:\Upiita\6to\Patrones\Objetos\guitarra_voz\guitarra.wav")
t2=np.arange(len(guitarra))/float(muestreo2)
guitarra=guitarra/(2.**15)
# plt.close('all')
plt.figure(figsize=(18,4))
plt.plot(t2,guitarra,label='senal guitarra',color='g')
plt.legend()

# PlaySound(r"D:\Upiita\6to\Patrones\Objetos\guitarra_voz\guitarra.wav",SND_FILENAME|SND_ASYNC)

min_len=min(len(voz),len(guitarra))
voz=voz[:min_len]
guitarra=guitarra[:min_len]
t1=t1[:min_len]
target=voz + guitarra

# plt.close('all')
plt.figure(figsize=(18,4))
plt.plot(t1,target,label='Target',color='r')
plt.legend()

w=np.array([-0.5,0.5,0.3])
b=-0.4
patron=np.zeros((3,1))
salida=np.zeros((len(guitarra),1))
alpha=0.1

#Entrenamiento de neurona ADALINE

for t in range(len(guitarra)):
    if t==1:
        patron[0]=guitarra[t]
    elif t==2:
        patron[0]=guitarra[t]
        patron[1]=guitarra[t-1]
    else:
        patron[0]=guitarra[t]
        patron[1]=guitarra[t-1]
        patron[2]=guitarra[t-2]

    out=np.dot(w,patron.flatten())+b
    error=target[t]-out
    salida[t]=error
    w=w+2*(alpha*error*patron.flatten())
    b=b+(2*alpha*error)

plt.figure(figsize=(18,4))
plt.plot(t1,salida,label='Salida de ADALINE',color='m')
plt.legend()

neurona=salida*(2.**15)
neurona=np.array(neurona,dtype=np.int16)
wavfile.write("recupera.wav",muestreo1,neurona)

plt.show()


#Modelo de autocorrelación para comparar entre señales distintas, con lpc método de yule walker para hacer predicciones