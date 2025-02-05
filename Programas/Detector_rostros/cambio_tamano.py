#Con las imágenes en el mismo formato, pasar a imágemes de tamaño 256x256
#<=== Aquí manipulamos las imagenes
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
import pandas as pd

# Loading 
direccion = "D:\Upiita\6to\Patrones\Programas\Detector_rostros\Imagenes"

lista_archivos = os.listdir(direccion)
lista_archivos = lista_archivos[0:500]

listado=[]

for ii in lista_archivos:
    listado.append("C:/Users/mcs54190/Documents/Reconocimiento_Patrones/AlexNET/Datospublicacion/training/cats/"+ii)
    ima = io.imread(listado[-1])
    ima = ( transform.resize(ima, (200, 250))*255.0 ).astype('uint8')
    io.imsave( listado[-1], ima )

plt.figure()
plt.imshow(ima)

# df = pd.read_csv("C:/Users/mcs54190/Documents/Reconocimiento_Patrones/AlexNET/Datospublicacion/etiquetados.csv")
# df = df[['nombre', 'clase']]