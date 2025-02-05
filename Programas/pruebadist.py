from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

nombres_clases = ['Adelie Penguin', 'Chinstrap penguin', 'Gentoo penguin']
train_data = pd.read_csv(r'D:\\Upiita\6to\Patrones\penguins_training.csv')
test_data = pd.read_csv(r'D:\\Upiita\6to\Patrones\penguins_testing.csv')

species = {'Adelie Penguin (Pygoscelis adeliae)': 'blue',
           'Chinstrap penguin (Pygoscelis antarctica)': 'green',
           'Gentoo penguin (Pygoscelis papua)': 'red'}

clas1 = train_data[train_data['Species'] == 'Adelie Penguin (Pygoscelis adeliae)']
map1 = [clas1['Culmen Length (mm)'], clas1['Culmen Depth (mm)'], clas1['Flipper Length (mm)']]
cla1 = np.mean(map1, 1)

clas2 = train_data[train_data['Species'] == 'Chinstrap penguin (Pygoscelis antarctica)']
map2 = [clas2['Culmen Length (mm)'], clas2['Culmen Depth (mm)'], clas2['Flipper Length (mm)']]
cla2 = np.mean(map2, 1)

clas3 = train_data[train_data['Species'] == 'Gentoo penguin (Pygoscelis papua)']
map3 = [clas3['Culmen Length (mm)'], clas3['Culmen Depth (mm)'], clas3['Flipper Length (mm)']]
cla3 = np.mean(map3, 1)


def distancia_mahalanobis(dato, promedio, covarianza):
    diff = dato - promedio
    d_mahalanobis = np.sqrt(np.dot(np.dot(diff.T, np.linalg.inv(covarianza)), diff))
    return d_mahalanobis

def distancia_coseno(dato, promedio):
    result = np.dot(dato, promedio)/(np.linalg.norm(dato)*np.linalg.norm(promedio))
    return result

resultados = []
contador = 1  

caracteristicas = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)']
covarianza = np.cov(train_data[caracteristicas].T)

for specie, color in species.items():
    subset = test_data[test_data['Species'] == specie]
    
    dist_euclidiana_1 = np.sqrt((subset['Culmen Length (mm)'] - cla1[0])**2 + (subset['Culmen Depth (mm)'] - cla1[1])**2 + (subset['Flipper Length (mm)'] - cla1[2])**2)
    dist_euclidiana_2 = np.sqrt((subset['Culmen Length (mm)'] - cla2[0])**2 + (subset['Culmen Depth (mm)'] - cla2[1])**2 + (subset['Flipper Length (mm)'] - cla2[2])**2)
    dist_euclidiana_3 = np.sqrt((subset['Culmen Length (mm)'] - cla3[0])**2 + (subset['Culmen Depth (mm)'] - cla3[1])**2 + (subset['Flipper Length (mm)'] - cla3[2])**2)
    
    distancias_euclidiana = np.vstack((dist_euclidiana_1, dist_euclidiana_2, dist_euclidiana_3)).T
    
    dist_mahalanobis_1 = subset[caracteristicas].apply(lambda x: distancia_mahalanobis(x.to_numpy(), cla1, covarianza), axis=1)
    dist_mahalanobis_2 = subset[caracteristicas].apply(lambda x: distancia_mahalanobis(x.to_numpy(), cla2, covarianza), axis=1)
    dist_mahalanobis_3 = subset[caracteristicas].apply(lambda x: distancia_mahalanobis(x.to_numpy(), cla3, covarianza), axis=1)
    
    distancias_mahalanobis = np.vstack((dist_mahalanobis_1, dist_mahalanobis_2, dist_mahalanobis_3)).T
    
    dist_coseno_1 = subset[caracteristicas].apply(lambda x: distancia_coseno(x.to_numpy(), cla1),axis = 1)
    dist_coseno_2 = subset[caracteristicas].apply(lambda x: distancia_coseno(x.to_numpy(), cla2), axis = 1)
    dist_coseno_3 = subset[caracteristicas].apply(lambda x: distancia_coseno(x.to_numpy(), cla3),axis = 1)

    distancias_coseno = np.vstack((dist_coseno_1,dist_coseno_2,dist_coseno_3)).T
       
    
    indices_minimos_euclidiana = np.argmin(distancias_euclidiana, axis=1)
    valores_minimos_euclidiana = np.min(distancias_euclidiana, axis=1)
    
    indices_minimos_mahalanobis = np.argmin(distancias_mahalanobis, axis=1)
    valores_minimos_mahalanobis = np.min(distancias_mahalanobis, axis=1)
    
    indices_minimos_coseno = np.argmax(distancias_coseno, axis=1)
    valores_minimos_coseno = np.max(distancias_coseno, axis=1)
    
    clases_cercanas_euclidiana = [nombres_clases[idx] for idx in indices_minimos_euclidiana]
    clases_cercanas_mahalanobis = [nombres_clases[idx] for idx in indices_minimos_mahalanobis]
    clases_cercanas_coseno = [nombres_clases[idx] for idx in indices_minimos_coseno]
    
    for i in range(len(subset)):
        resultados.append({
            'No.': contador,
            'Clase inicial': specie,
            '(Euclidiana)': clases_cercanas_euclidiana[i],
            'Distancia Mínima (Euclidiana)': valores_minimos_euclidiana[i],
            '(Mahalanobis)': clases_cercanas_mahalanobis[i],
            'Distancia Mínima (Mahalanobis)': valores_minimos_mahalanobis[i],
            '(Coseno)':clases_cercanas_coseno[i],
            'Mayor similitud':valores_minimos_coseno[i],
        })
        contador += 1

# Mostrar los resultados en una tabla
tabla_resultados = pd.DataFrame(resultados)

num_rows = len(tabla_resultados)
chunk_size = 35 

for start_row in range(0, num_rows, chunk_size):
    end_row = min(start_row + chunk_size, num_rows)
    chunk = tabla_resultados.iloc[start_row:end_row]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=chunk.values, colLabels=chunk.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
plt.show()