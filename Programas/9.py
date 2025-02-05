import matplotlib.pyplot as plt
from skimage import data, io, color
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, convex_hull_image
from skimage.morphology import disk
 
# Leer la imagen
imagen = io.imread(r"D:\Upiita\6to\Patrones\Objetos\Placas base de datos\img5.jpeg")
gris = color.rgb2gray(imagen)
 
plt.close('all')
plt.subplot(1, 2, 1)
plt.imshow(imagen)
plt.title('Imagen Original')
plt.axis('off')  # Ocultar los ejes
 
# Mostrar el histograma en el segundo subgráfico
plt.subplot(1, 2, 2)
plt.imshow(gris, cmap='gray')
plt.title('Imagen en escala de grises')
plt.axis('off')  # Ocultar los ejes
 
# Mostrar el gráfico
plt.tight_layout()
# plt.show()
 
binario = gris < 0.5
suma_col = np.sum(binario, axis=0)
 
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.imshow(binario, cmap="gray")
plt.title('Imagen Binaria')
plt.axis('off')  # Ocultar los ejes
 
# Mostrar el histograma en el segundo subgráfico
plt.subplot(1, 2, 2)
plt.plot(suma_col)
plt.title('Suma de columnas')
 
# Mostrar el gráfico
plt.tight_layout()
# plt.show()
 
 
digito_01 = binario[:, 460:550]
digito_02 = binario[:, 572:663]
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.imshow(digito_01, cmap='gray')
plt.title('Primer digito')
plt.axis('off')  # Ocultar los ejes
 
# Mostrar el histograma en el segundo subgráfico
plt.subplot(1, 2, 2)
plt.imshow(digito_02, cmap='gray')
plt.title('Segundo digito')
plt.axis('off')  # Ocultar los ejes
 
# Mostrar el gráfico
plt.tight_layout()
# plt.show()
 
estructura = disk(5) #tamaño del radio del disco
ero = closing(digito_01, estructura)
estructura = disk(5)
dil = opening(digito_02, estructura)
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.imshow(ero, cmap='gray')
plt.title('Primer digito')
plt.axis('off')  # Ocultar los ejes
 
# Mostrar el histograma en el segundo subgráfico
plt.subplot(1, 2, 2)
plt.imshow(dil, cmap='gray')
plt.title('Segundo digito')
plt.axis('off')  # Ocultar los ejes
 
# Mostrar el gráfico
plt.tight_layout()
# plt.show()
 
hull1 = convex_hull_image(dil == 0) #(dil == 1) la parte del convex hull intenta rellenar todos los espacios vacios concatenando todos los blancos
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.imshow(dil, cmap='gray')
plt.title('Primer digito')
plt.axis('off')  # Ocultar los ejes
 
# Mostrar el histograma en el segundo subgráfico
plt.subplot(1, 2, 2)
plt.imshow(hull1, cmap='gray')
plt.title('Segundo digito')
plt.axis('off')  # Ocultar los ejes
 
# Mostrar el gráfico
plt.tight_layout()
# plt.show()
 
# m00 = 0
# m01 = 0
# m10 = 0
# for y in range (dil.shape[0]):
#     for x in range (dil.shape[1]):
#         m00 = ( (x)**(0) * (y)**(0) * dil[y,x] ) + m00
#         m10 = ( (x)**(1) * (y)**(0) * dil[y,x] ) + m10
#         m01 = ( (x)**(0) * (y)**(1) * dil[y,x] ) + m01
 
# cen_x = m01/m00
# cen_y = m10/m00
 
# mu00 = 0
# mu10 = 0
# mu01 = 0
# mu20 = 0
# mu02 = 0
# for y in range (dil.shape[0]):
#     for x in range (dil.shape[1]):
#         mu00 = ( (x-cen_x)**(0) * (y-cen_y)**(0) * dil[y,x] ) + mu00
#         mu10 = ( (x-cen_x)**(1) * (y-cen_y)**(0) * dil[y,x] ) + mu10
#         mu01 = ( (x-cen_x)**(0) * (y-cen_y)**(1) * dil[y,x] ) + mu01
#         mu20 = ( (x-cen_x)**(2) * (y-cen_y)**(0) * dil[y,x] ) + mu20
#         mu02 = ( (x-cen_x)**(0) * (y-cen_y)**(2) * dil[y,x] ) + mu02

# eta02=mu02/(mu00**2)
# eta20=mu20/(mu00**2)
# fi1=eta02+eta20
# print(fi1)

perfil=[]

for y in range (dil.shape[0]):
    for x in range (dil.shape[1]):
        if dil[y,x]==1:
            perfil.append(x)
            break

for y in range (dil.shape[0]):
    for x in range (dil.shape[1]-1,1,-1):
        if dil[y,x]==1:
            perfil.append(x)
            break

#np.corrcoef(base,entrada) da el coeficiente de correlación correspondiente de entre una base y una entrada al programa [ semejanza entre x x , semejanza entre x y ; semejanza entre y x, semejanza entre y y]

plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.plot(perfil)
plt.title('Primer digito')
 
# Mostrar el histograma en el segundo subgráfico
plt.subplot(1, 2, 2)
plt.imshow(hull1, cmap='gray')
plt.title('Segundo digito')

plt.show()

#NN perceptron multicapa muy sencilla, sklearn. Pytorch más compleja, tensor todavía más compleja