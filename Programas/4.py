import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

train_data=pd.read_csv(r'D:\\Upiita\6to\Reconocimiento de Patrones\penguins_training.csv')

fig = plt.figure(figsize=(10,8))
ax=fig.add_subplot(111,projection='3d')
species={'Adelie Penguin (Pygoscelis adeliae)':'blue',
         'Chinstrap penguin (Pygoscelis antarctica)':'green',
         'Gentoo penguin (Pygoscelis papua)':'red'}

for specie,color in species.items():
    subset=train_data[train_data['Species']==specie]
    ax.scatter(subset['Culmen Length (mm)'],subset['Culmen Depth (mm)'], subset['Flipper Length (mm)'],color=color,label=specie,alpha=0.6)

ax.set_xlabel('Culmen Length(mm)')
ax.set_ylabel('Culmen Depth(mm)')
ax.set_zlabel('Flipper Length (mm)')
plt.title('Largo vs Alto del Pico vs Largo de la aleta')
plt.legend()
plt.grid(True)

plt.show()

##Hacer excel con: Muestra | Clase inicial(del excel test) | Clase otorgada (Por el programa) | distancia euclidiana min
# Práctica 1. Si realmente se pueden separar machos de hembras, cual característica, solamente dos clases 
