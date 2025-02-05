import numpy as np
import matplotlib.pyplot as plt
from skimage import data,io,color,morphology

plt.close()
imagen=io.imread(r'D:\Upiita\6to\Patrones\Objetos\placa.jpg')

plt.figure(0)
plt.imshow(imagen)

gris=color.rgb2gray(imagen)
plt.figure(1)
plt.imshow(gris,cmap="gray")

binaria=(gris<0.5)*255.0

plt.figure(2)
plt.imshow(binaria,cmap="gray")

sumaf=np.sum(binaria,axis=1) #suma sobre la variable binaria en axis=0 es decir columnas, si es axis=1, será filas 
plt.figure(3)
plt.plot(sumaf)

recorte1=binaria[818:1290,:] #Recorte por filas según lospuntos más bajos del plt anterior
plt.figure(4)
plt.imshow(recorte1,cmap="gray")

sumac=np.sum(recorte1,axis=0) #suma sobre la variable binaria en axis=0 es decir columnas, si es axis=1, será filas 
plt.figure(5)
plt.plot(sumac)

#Porvisualización de la gráfica
num1=recorte1[:,1165:1306] #8
num2=recorte1[:,1346:1479] #0
plt.figure(6)
plt.imshow(num1,cmap="gray")
plt.figure(7)
plt.imshow(num2,cmap="gray")

#Erosion y después dilatación = opening en viceversa sería closing

# num1=morphology.erosion(num1)
# num2=morphology.erosion(num2)

#Revisar como se puede modificar el elemento estructurante matriz del filtro) y comando fill en skimage
# num1=morphology.opening(num1)
# num2=morphology.opening(num2)
num1=morphology.area_opening(num1)
num2=morphology.area_opening(num2)
plt.figure(8)
plt.imshow(num1,cmap="gray")
plt.figure(9)
plt.imshow(num2,cmap="gray")



plt.show()