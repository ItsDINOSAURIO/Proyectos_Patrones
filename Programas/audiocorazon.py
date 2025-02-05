from scipy.io import wavfile
import matplotlib.pyplot as plt
from winsound import PlaySound,SND_FILENAME,SND_ASYNC
import numpy as np
import time

muestreo_01,corazon=wavfile.read(r"D:\Upiita\6to\Patrones\Objetos\audiocorazon.wav")
print(len(corazon))
corazon=corazon[46167444:]
print(len(corazon))
t1=np.arange(len(corazon))/float(muestreo_01)

plt.close("all")
plt.figure(figsize=(18,4))
plt.plot(t1,corazon,label='Senal Corazon',color='b')
plt.xlabel('Tiempo(s)')
plt.ylabel('Amplitud')
plt.title('Señal Original')
plt.legend()

PlaySound(r"D:\Upiita\6to\Patrones\Objetos\audiocorazon.wav",SND_ASYNC)

plt.show()