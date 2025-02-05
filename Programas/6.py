import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#ctrl+k+c comentar
#ctrl+k+u quitar comentario
# Práctica 1. Si realmente se pueden separar machos de hembras, cual característica = culmen depth, solamente dos clases 
train_data=pd.read_csv(r'D:\\Upiita\6to\Patrones\penguins_training.csv')

Sex={'MALE':'green', 'FEMALE':'magenta'}

#clase_1
val=train_data[train_data['Sex']=='MALE'] 
car=[val['Culmen Length (mm)'],val['Culmen Depth (mm)'],val['Flipper Length (mm)']]
clase1=np.mean(car,1) 

#clase_2
val=train_data[train_data['Sex']=='FEMALE'] 
car=[val['Culmen Length (mm)'],val['Culmen Depth (mm)'],val['Flipper Length (mm)']]
clase2=np.mean(car,1)

#Matriz de covarianza
car=train_data[['Culmen Length (mm)','Culmen Depth (mm)','Flipper Length (mm)']]
cov=car.cov()

salida=[]
mue=1
test_data=pd.read_csv(r'D:\\Upiita\6to\Patrones\penguins_testing.csv')
for sex,color in Sex.items():
    val=test_data[test_data['Sex']==sex]
   
    for i, fila in val.iterrows():
        Clase_inicial=fila['Sex']
        dato = [fila['Culmen Length (mm)'],fila['Culmen Depth (mm)'],fila['Flipper Length (mm)']]
        dist1 = np.sqrt(np.dot(np.dot((dato - clase1).T, np.linalg.inv(cov)), (dato - clase1)))
        dist2 = np.sqrt(np.dot(np.dot((dato - clase2).T, np.linalg.inv(cov)), (dato - clase2)))
        Distancia=np.array([dist1,dist2])
        minimo=np.argmin(Distancia)
        valor_minimo=Distancia[minimo]
        if minimo ==0 and valor_minimo<50:
            Clase_otorgada='MALE'
        elif minimo ==1 and valor_minimo<50:
            Clase_otorgada='FEMALE'
        else: Clase_otorgada='Desconocida'

        salida.append([mue,Clase_inicial,Clase_otorgada,valor_minimo])
        mue+=1

excel = pd.DataFrame(salida, columns=['Muestra', 'Clase inicial', 'Clase otorgada', 'Distancia de Mahalanobis mínima'])
print(excel)
print (cov)
#excel.to_excel('resultados_penguins_sex.xlsx')


fig = plt.figure(figsize=(10,8))
ax=fig.add_subplot(111,projection='3d')
         
for sex,color in Sex.items():
    subset=train_data[train_data['Sex']==sex]
    ax.scatter(subset['Culmen Length (mm)'],subset['Culmen Depth (mm)'], subset['Flipper Length (mm)'],color=color,label=sex,alpha=0.6)

ax.set_xlabel('Culmen Length(mm)')
ax.set_ylabel('Culmen Depth(mm)')
ax.set_zlabel('Flipper Length (mm)')
plt.title('Largo vs Alto del Pico vs Largo de la aleta')
plt.legend()
plt.grid(True)

plt.show()

##Hacer excel con: Muestra | Clase inicial(del excel test) | Clase otorgada (Por el programa) | distancia euclidiana min | distancia mahalanobis min | coseno min
#Hacer con distancia de mahalanobis matriz Mu(3,150) Matriz de covarianza : Cov=Mu'*Mu comparar carecterísticas para obtener matriz 3x3

#Distancia coseno

salidac=[]
muec=1

for sex,color in Sex.items():
    val=test_data[test_data['Sex']==sex]
   
    for i, fila in val.iterrows():
        Clase_inicial=fila['Sex']
        dato = [fila['Culmen Length (mm)'],fila['Culmen Depth (mm)'],fila['Flipper Length (mm)']]
        dist1 = np.dot(dato,clase1)/(np.linalg.norm(dato)*np.linalg.norm(clase1))
        dist2 = np.dot(dato,clase2)/(np.linalg.norm(dato)*np.linalg.norm(clase2))
        Distancia=np.array([dist1,dist2])
        minimo=np.argmin(Distancia)
        valor_minimo=Distancia[minimo]
        if minimo ==0 and valor_minimo<50:
            Clase_otorgada='MALE'
        elif minimo ==1 and valor_minimo<50:
            Clase_otorgada='FEMALE'
        else: Clase_otorgada='Desconocida'

        salidac.append([muec,Clase_inicial,Clase_otorgada,valor_minimo])
        muec+=1

excelc = pd.DataFrame(salidac, columns=['Muestra', 'Clase inicial', 'Clase otorgada', 'Distancia Coseno mínima'])
print(excelc)