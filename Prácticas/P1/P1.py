import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Ctrl+K+C para comentar
# Ctrl+K+U para descomentar

# Práctica 1. Si realmente se pueden separar machos de hembras, cual característica = culmen depth, solamente dos clases 
train_data = pd.read_csv(r'D:\\Upiita\6to\Patrones\penguins_training.csv')
test_data = pd.read_csv(r'D:\\Upiita\6to\Patrones\penguins_testing.csv')

Sex = {'MALE': 'green', 'FEMALE': 'magenta'}
species = {'Adelie Penguin (Pygoscelis adeliae)': 'blue',
           'Chinstrap penguin (Pygoscelis antarctica)': 'green',
           'Gentoo penguin (Pygoscelis papua)': 'red'}

# Matriz de covarianza
car = train_data[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)']]
cov = car.cov()

def carac(val):
    car = val[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)']].values
    prom = np.mean(car, axis=0)
    return prom

def eucl(dato, clases):
    d = []
    for fila in clases:
        dist = np.sqrt(np.sum((dato - fila) ** 2))
        d.append(dist)
    return np.array(d)

def mahal(dato, clases):
    d = []
    inv_cov = np.linalg.inv(cov)
    for fila in clases:
        diff = dato - fila
        dist = np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))
        d.append(dist)
    return np.array(d)

def cos(dato, clases):
    d = []
    for fila in clases:
        dist = np.dot(dato, fila) / (np.linalg.norm(dato) * np.linalg.norm(fila))
        d.append(dist)
    return np.array(d)

def che(dato, clases):
    d = []
    for fila in clases:
        dist=np.max(np.abs(dato-fila))
        d.append(dist)
    return np.array(d)


def mind(im, vm,cat):
    if cat == 'Sex':
        if im == 0 and vm < 30:
            return 'MALE'
        elif im == 1 and vm < 30:
            return 'FEMALE'
        else:
            return 'Desconocida'
    elif cat== 'Species':
        if im == 0 and vm < 30:
            return 'Adelie Penguin (Pygoscelis adeliae)'
        elif im == 1 and vm < 30:
            return 'Chinstrap penguin (Pygoscelis antarctica)'
        elif im == 2 and vm < 30:
            return 'Gentoo penguin (Pygoscelis papua)'
        else:
            return 'Desconocida'

def df(salida):
    tr=pd.DataFrame(salida, columns=['Muestra', 'Clase inicial', 'Clase otorgada (euclidiana)', 'Distancia Euclidiana mínima', 'Clase otorgada (Mahalanobis)', 'Distancia de Mahalanobis mínima', 'Clase otorgada (Cosenos)', 'Mayor similitud por Cosenos','Clase otorgada (Chebyshev)','Distancia de Chebyshev mínima'])
    nf = len(tr)
    tch = 35 

    for fi in range(0, nf, tch):
        ff = min(fi + tch, nf)
        ch = tr.iloc[fi:ff]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=ch.values, colLabels=ch.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1.2, 1.5)


'''Para Sexo'''
# clase_1
val = train_data[train_data['Sex'] == 'MALE'] 
clase1 = carac(val) 

# clase_2
val = train_data[train_data['Sex'] == 'FEMALE'] 
clase2 = carac(val)

clases_s = np.array([clase1, clase2])


'''Para Especie'''
# clase_1
val = train_data[train_data['Species'] == 'Adelie Penguin (Pygoscelis adeliae)'] 
clase1 = carac(val) 

# clase_2
val = train_data[train_data['Species'] == 'Chinstrap penguin (Pygoscelis antarctica)'] 
clase2 = carac(val)

# clase_3
val = train_data[train_data['Species'] == 'Gentoo penguin (Pygoscelis papua)'] 
clase3 = carac(val)

clases_e = np.array([clase1, clase2, clase3])

'''Para Sexo'''
salida = []
mue = 1

for sex, color in Sex.items():
    val = test_data[test_data['Sex'] == sex]
   
    for i, fila in val.iterrows():
        ci = fila['Sex']
        dato = [fila['Culmen Length (mm)'], fila['Culmen Depth (mm)'], fila['Flipper Length (mm)']]

        # Euclidiana
        diste = eucl(dato, clases_s)
        imin = np.argmin(diste)
        dm_e = diste[imin]
        co_e = mind(imin, dm_e,'Sex')

        # Mahalanobis
        distm = mahal(dato, clases_s)
        imin = np.argmin(distm)
        dm_m = distm[imin]
        co_m = mind(imin, dm_m,'Sex')

        # Coseno
        distc = cos(dato, clases_s)
        imin = np.argmax(distc)
        dm_c = distc[imin]  
        co_c = mind(imin, dm_c,'Sex') 

        # Chebyshev
        distch = che(dato, clases_s)
        imin = np.argmin(distch)
        dm_ch = distch[imin]  
        co_ch = mind(imin, dm_ch,'Sex')   

        salida.append([mue, ci, co_e, dm_e, co_m, dm_m, co_c, dm_c,co_ch,dm_ch])
        mue += 1

df1 = df(salida)
#print(df1)

'''Para Especies'''
salida = []
mue = 1

for specie, color in species.items():
    val = test_data[test_data['Species'] == specie]
   
    for i, fila in val.iterrows():
        ci = fila['Species']
        dato = [fila['Culmen Length (mm)'], fila['Culmen Depth (mm)'], fila['Flipper Length (mm)']]

        # Euclidiana
        diste = eucl(dato, clases_e)
        imin = np.argmin(diste)
        dm_e = diste[imin]
        co_e = mind(imin, dm_e,'Species')

        # Mahalanobis
        distm = mahal(dato, clases_e)
        imin = np.argmin(distm)
        dm_m = distm[imin]
        co_m = mind(imin, dm_m,'Species')

        # Coseno
        distc = cos(dato, clases_e)
        imin = np.argmax(distc)
        dm_c = distc[imin]  
        co_c = mind(imin, dm_c,'Species')    

        # Chebyshev
        distch = che(dato, clases_e)
        imin = np.argmin(distch)
        dm_ch = distch[imin]  
        co_ch = mind(imin, dm_ch,'Species')   

        salida.append([mue, ci, co_e, dm_e, co_m, dm_m, co_c, dm_c,co_ch,dm_ch])
        mue += 1

df2 = df(salida)
#print(df2)

plt.show()