import numpy as np
import matplotlib.pyplot as plt
from skimage import data,io

imagen=io.imread(r'D:\\Upiita\6to\Reconocimiento de Patrones\Objetos\1.jpg')
#imagen=data.astronaut()
#imagen=data.chelsea()
#imagen=data.camera()

Parte=imagen[1500:1800,500:800,:]
plt.figure(0)
plt.imshow(imagen)
plt.figure(1)
plt.imshow(Parte)

hr=np.zeros(256) #Histograma rojo
hv=np.zeros(256) #Histograma verde
ha=np.zeros(256) #Histograma azul

filas =  imagen.shape[0]
columnas=imagen.shape[1]
capas=imagen.shape[2]

for i in range (0, filas, 1):
    for j in range (0,columnas,1):
        PosR=imagen[i,j,0]
        PosV=imagen[i,j,1]
        PosA=imagen[i,j,2]
        hr[PosR]=hr[PosR]+1
        hv[PosV]=hv[PosV]+1
        ha[PosA]=ha[PosA]+1

plt.figure(2)
plt.plot(hr)
plt.figure(3)
plt.plot(hv)
plt.figure(4)
plt.plot(ha)

ax = plt.axes(projection="3d")

# Extraer coordenadas para el gráfico 3D
x = np.arange(Parte.shape[0])
y = np.arange(Parte.shape[1])
x, y = np.meshgrid(x, y)
z = Parte[:, :, 0].flatten()

ax.scatter(x.flatten(), y.flatten(), z, c=Parte.reshape(-1, 3)/255.0) #c es para darle el color rgb

plt.show()