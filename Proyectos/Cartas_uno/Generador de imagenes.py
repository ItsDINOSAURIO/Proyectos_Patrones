import matplotlib.pyplot as plt
from skimage import data, io, color
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, convex_hull_image
from skimage.morphology import disk
from skimage.transform import resize

#Funciones 

# Función para calcular la correlación
def correlacion(recorte, clases_p):
    correlaciones = []
    for clase in clases_p:
        corr = np.corrcoef(recorte, clase)[0, 1] 
        correlaciones.append(corr)
    return np.array(correlaciones)

# Función para calcular el perfil de una imagen binarizada
#Función para el cálculo del perfil
def perfil(imagen):
    sil = []  
    for y in range(imagen.shape[0]):
        for x in range(imagen.shape[1]):
            if imagen[y, x] == True:
                sil.append(x)
                break
    for y in range(imagen.shape[0]):
        for x in range(imagen.shape[1]-1, 1, -1):
            if imagen[y, x] == True:
                sil.append(x)
                break
#    if len(sil) < 80:
#        sil.extend([0] * (80 - len(sil)))
#    elif len(sil) > 80:
#        sil = sil[:80]
    return sil

# Leer la imagen
imagen = io.imread(r'C:\Users\erikn\Documents\School\7mo semestre\Reconocimiento de patrones\Cartas_uno\Comodin\+4.jpg')
imagen = resize(imagen, (700, 400), anti_aliasing=False)  
gris = color.rgb2gray(imagen)

binario = gris > 0.98
recorte = binario[6:160, 0:160]
estructura = disk(1)
ero = closing(recorte, estructura)

perfil1 = perfil(ero)   

plt.figure(figsize=(8, 6))
plt.subplot(1, 3, 1)
plt.imshow(binario, cmap="gray")
plt.title('Imagen Original')
plt.axis('off')  # Ocultar los ejes
plt.subplot(1, 3, 2)
plt.imshow(ero, cmap='gray')
plt.title('Primer digito')
plt.axis('off')  # Ocultar los ejes 
plt.subplot(1, 3, 3)
plt.plot(perfil1)
plt.title('Primer digito')

plt.show()