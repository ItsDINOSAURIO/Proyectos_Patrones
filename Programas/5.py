import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_data=pd.read_csv(r'D:\\Upiita\6to\Patrones\penguins_training.csv')

species={'Adelie Penguin (Pygoscelis adeliae)':'blue',
         'Chinstrap penguin (Pygoscelis antarctica)':'green',
         'Gentoo penguin (Pygoscelis papua)':'red'}

#clase_1
val=train_data[train_data['Species']=='Adelie Penguin (Pygoscelis adeliae)'] 
car=[val['Culmen Length (mm)'],val['Culmen Depth (mm)'],val['Flipper Length (mm)']]
clase_obj1=np.mean(car,1) 

#clase_2
val=train_data[train_data['Species']=='Chinstrap penguin (Pygoscelis antarctica)'] 
car=[val['Culmen Length (mm)'],val['Culmen Depth (mm)'],val['Flipper Length (mm)']]
clase_obj2=np.mean(car,1)

#clase_3
val=train_data[train_data['Species']=='Gentoo penguin (Pygoscelis papua)'] 
car=[val['Culmen Length (mm)'],val['Culmen Depth (mm)'],val['Flipper Length (mm)']]
clase_obj3=np.mean(car,1)

salida=[]

test_data=pd.read_csv(r'D:\\Upiita\6to\Patrones\penguins_testing.csv')
for specie,color in species.items():
    val=test_data[test_data['Species']==specie]
    for i, fila in val.iterrows():
        Clase_inicial=fila['Species']
        dato = [fila['Culmen Length (mm)'],fila['Culmen Depth (mm)'],fila['Flipper Length (mm)']]
        dist1= np.sqrt((dato[0]-clase_obj1[0])**2+(dato[1]-clase_obj1[1])**2+(dato[2]-clase_obj1[2])**2)
        dist2= np.sqrt((dato[0]-clase_obj2[0])**2+(dato[1]-clase_obj2[1])**2+(dato[2]-clase_obj2[2])**2)
        dist3= np.sqrt((dato[0]-clase_obj3[0])**2+(dato[1]-clase_obj3[1])**2+(dato[2]-clase_obj3[2])**2)
        Distancia=np.array([dist1,dist2,dist3])
        minimo=np.argmin(Distancia)
        valor_minimo=Distancia[minimo]
        if minimo ==0 and valor_minimo<50:
            Clase_otorgada='Adelie Penguin (Pygoscelis adeliae)'
        elif minimo ==1 and valor_minimo<50:
            Clase_otorgada='Chinstrap penguin (Pygoscelis antarctica)'
        elif minimo ==2 and valor_minimo<50:
            Clase_otorgada='Gentoo penguin (Pygoscelis papua)'
        else: Clase_otorgada='Desconocida'

        salida.append([i+1,Clase_inicial,Clase_otorgada,valor_minimo])

excel = pd.DataFrame(salida, columns=['Muestra', 'Clase inicial', 'Clase otorgada', 'Distancia euclidiana mínima'])
print(excel)
#excel.to_excel('resultados_penguins.xlsx')