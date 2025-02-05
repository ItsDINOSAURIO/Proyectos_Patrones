#Distancia de Mahalanobis
from skimage import io , transform
from numpy import linalg # <=== subpaquete de algebra lineal
import numpy as np
import matplotlib . pyplot as plt

plt.close('all')

# === Funci ón para el cá lculo de la distancia de Mahalanobis
# Parametros de entrada , el pí xel a procesar

Ima = io . imread (r'D:\Upiita\6to\Patrones\sky_01.jpg') # <=== Lectura de la imagen
# === la imagen es re - escalda a 640 x 480 pixeles
#Ima = transform.resize( Ima , [480 , 640 , 3])

# === Aqui se designa la cantidad de muestras a recopilar
pixeles = int( input ('Cuantas Desea Recopilar ? : ') )
plt.figure(1)
plt.imshow(Ima)

# === Muestras de la clase rojo5

posCielo = np . int32 ( plt . ginput ( pixeles ) )
valCielo = Ima [ posCielo [: , 1] , posCielo [: , 0]]

cova = np.cov( valCielo.T , ddof = 0) # ddof covarianza de los datos en fila con 1, en 0 por columna
prom = np.mean( valCielo , axis = 0) # axis = 0 valores por columna
#print(cova)
#la covarianza de los pí xeles de entrenamiento
def distancia ( pixel , promedio , covarianza ) :
    d1 = ( pixel - prom ).dot(np.linalg.inv( covarianza ))
    d2 = d1.dot(( pixel - prom ).T)
    d3 = np.sqrt (d2)
    return d3

#<============================================================== >#
filas = Ima . shape [0] #=== Numero de filas
columnas = Ima . shape [1] #=== Numero de columnas
capa = Ima . shape [2] #=== Canal de color

# === Imagen que contiene el resultado de la clasificacion
salida = np . zeros (( filas , columnas , capa ) )
for i in range ( filas ) :
    for j in range ( columnas ) :
        pix = Ima [i ,j ,:]
        dis = distancia ( pix , prom , cova )
        if dis > 3: # <=== Seleccion de umbral
            salida [i , j , 0] = 1
            salida [i , j , 1] = 1
            salida [i , j , 2] = 1

horizonte = Ima * salida # <=== Multiplicacion punto a punto

# Definir una figura y los ejes para el grafico
fig , axs = plt . subplots (1 , 3)

# Mostrar la imagen original en el primer subplot
axs [0]. imshow ( Ima )
axs [0]. set_title ('Imagen Original ')

# Mostrar la imagen binaria en el segundo subplot
axs [1]. imshow ( salida )
axs [1]. set_title ('Imagen Binaria')

# Mostrar la imagen del horizonte en el tercer subplot
axs [2]. imshow ( horizonte )
axs [2]. set_title ('Extraccion del cielo')

# Ajustar el espacio entre los subplots para que no se superpongan
plt . tight_layout ()

# Mostrar el grafico
plt . show ()

#Distancia Coseno

