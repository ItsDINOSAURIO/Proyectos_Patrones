#Llamar bibliotecas
 
import numpy as np
import matplotlib.pyplot as plt
from skimage import data,io

imagen = data.astronaut()
imagen=io.imread(r'D:\\Upiita\6to\Reconocimiento de Patrones\Objetos\10.jpg') # 3, 9 10
parte=imagen[:,:,:]

r=parte[:,:,0].flatten() #flatten vuelve vector una matriz de datos
g=parte[:,:,1].flatten()
b=parte[:,:,2].flatten() 

fig = plt.figure(figsize=(16,8))

#1°plot imagen

ax1=fig.add_subplot(121)
ax1.imshow(parte)
ax1.set_title('Imagen inicial')
ax1.axis('off')

ax2=fig.add_subplot(122,projection='3d')
ax2.scatter(r,g,b,c=np.vstack([r,g,b]).T/255,s=1,marker='o')
ax2.set_xlim([0,255])
ax2.set_ylim([0,255])
ax2.set_zlim([0,255])
ax2.set_xlabel('Canal R')
ax2.set_ylabel('Canal G')
ax2.set_zlabel('Canal B')

cla1 = np.array([143,139,93]) #Plano
cla2 = np.array([203,103,85]) #Rojo
cla3 = np.array([85,164,72]) #Verde

filas,columnas,capas=imagen.shape

ni=np.zeros([filas,columnas,capas],dtype='int16')

for i in range(0,filas,1):
    for j in range(0,columnas,1):
        dato=imagen[i,j,:]
        dist1= np.sqrt((dato[0]-cla1[0])**2+(dato[1]-cla1[1])**2+(dato[2]-cla1[2])**2)
        dist2= np.sqrt((dato[0]-cla2[0])**2+(dato[1]-cla2[1])**2+(dato[2]-cla2[2])**2)
        dist3= np.sqrt((dato[0]-cla3[0])**2+(dato[1]-cla3[1])**2+(dato[2]-cla3[2])**2)
        d=np.array([dist1,dist2,dist3])
        min=np.argmin(d) #Da la posición del valor mínimo
        if min ==0:
            ni[i,j,:]=[255,255,255]
        elif min ==1:
            ni[i,j,:]=[255,0,0]
        elif min ==2:
            ni[i,j,:]=[0,255,0]

plt.figure(0)
plt.imshow(ni)


plt.show()