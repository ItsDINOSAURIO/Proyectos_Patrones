import numpy as np
import matplotlib.pyplot as plt
from skimage import data,io #from realiza simplemente la importación del losrecursos que se irán a utilizar, en lugar de la librería completa

imagen=io.imread(r'D:\\Upiita\6to\Reconocimiento de Patrones\Objetos\10.jpg') # 3, 9 10

plt.figure(0)
plt.imshow(imagen)

#clase_rojo
valores=int(input('¿Cuantas muestras quieres de rojo?: ')) 
pos=np.int32(plt.ginput(valores))
pixeles=imagen[pos[:,1],pos[:,0]]
clase_obj1=np.mean(pixeles,0) #El 1 indica un promedio de forma horizontal (por fila) en la matriz, para análisis de imágenes con matrices RGB, entonces se realiza un promedio por columnas con 0

#clase_verde
valores_01=int(input('¿Cuantas muestras quieres de verde?: ')) 
pos_01=np.int32(plt.ginput(valores_01))
pixeles_01=imagen[pos_01[:,1],pos_01[:,0]]
clase_obj2=np.mean(pixeles_01,0)

#clase_fondo
valores_02=int(input('¿Cuantas muestras del fondo?: '))
pos_02=np.int32(plt.ginput(valores_02))
pixeles_02=imagen[pos_02[:,1],pos_02[:,0]]
clase_fondo=np.mean(pixeles_02,0)

salida=np.zeros((imagen.shape[0],imagen.shape[1]))

for i in range(imagen.shape[0]):
    for j in range(imagen.shape[1]):
        dato = imagen[i,j,:]
        dist1= np.sqrt((dato[0]-clase_fondo[0])**2+(dato[1]-clase_fondo[1])**2+(dato[2]-clase_fondo[2])**2)
        dist2= np.sqrt((dato[0]-clase_obj1[0])**2+(dato[1]-clase_obj1[1])**2+(dato[2]-clase_obj1[2])**2)
        dist3= np.sqrt((dato[0]-clase_obj2[0])**2+(dato[1]-clase_obj2[1])**2+(dato[2]-clase_obj2[2])**2)
        Distancia=np.array([dist1,dist2,dist3])
        minimo=np.argmin(Distancia)
        valor_minimo=Distancia[minimo]
        if minimo ==0 and valor_minimo<50:
            salida[i,j]=255
        elif minimo ==1 and valor_minimo<50:
            salida[i,j]=200
        elif minimo ==2 and valor_minimo<50:
            salida[i,j]=150
        else: salida[i,j]=0 #Clase ruido 

plt.figure(1)
plt.imshow(salida)
plt.show()