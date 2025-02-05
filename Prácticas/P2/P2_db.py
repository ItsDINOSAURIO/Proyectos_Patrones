#El siguiente programa será dedicado únicamente para realizar la segmentación de las imágenes para su repartición en distintas carpetas.
import matplotlib.pyplot as plt
from skimage import io, color, measure
import numpy as np
from skimage.morphology import dilation, closing, area_closing, remove_small_objects, disk
from skimage.transform import resize
import os
import pandas as pd

def gbf(imagen):
    # Conversión a grises
    gris = color.rgb2gray(imagen)

    # Conversión a Binaria y filtrado
    estructura = disk(4.8)
    binario = (gris < 0.3)
    binario = closing(binario, estructura)
    binario = dilation(binario, estructura)
    binario = remove_small_objects(binario, min_size=100)
    binario = area_closing(binario, area_threshold=450)

    return binario

def momentos(imagen):
    mu = measure.moments(imagen)
    nu = measure.moments_normalized(mu)
    hu = measure.moments_hu(nu)  
    return momentos_hu.append(hu)

def perfil(imagen):
    sil = []  
    for y in range(imagen.shape[0]):
        for x in range(imagen.shape[1]):
            if imagen[y, x] == 255:
                sil.append(x)
                break

    for y in range(imagen.shape[0]):
        for x in range(imagen.shape[1]-1, 1, -1):
            if imagen[y, x] == 255:
                sil.append(x)
                break
    # plt.plot(sil)
    # plt.show()
    return perfiles.append(sil)

def imag_com(imagen):
    vector=imagen.flatten()
    vector=vector/np.max(vector)
    return vector

def df(metodo, nombre):
    df = pd.DataFrame(metodo, 
                      columns=[f'Caracteristica {i+1}' for i in range(metodo.shape[1])], 
                      index=clases_aux)
    df.index.name = 'Clases'

    # Guardar el DataFrame como CSV
    archivo_csv = os.path.join(carpeta_i, f'{nombre}.csv')
    df.to_csv(archivo_csv) 
    return df

# Elección y lectura de imágenes desde local
carpeta_i = r'D:\Upiita\6to\Patrones\Objetos\Placas base de datos\Numeros'
clases = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

clases_aux = []
momentos_hu = [] 
perfiles = []
vectores=[]

for clase in clases:
    ruta_c = os.path.join(carpeta_i, clase)
    imagenes = [f for f in os.listdir(ruta_c) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    for imagen in imagenes:
        ruta_i = os.path.join(ruta_c, imagen)
        imagen = io.imread(ruta_i)
        print(imagen)
        # Calcula los momentos de Hu y el perfil para cada imagen
        hu = momentos(imagen)
        perf = perfil(imagen)
        vec = imag_com(imagen) 
        
        vectores.append(vec)
        
      

        clases_aux.append(f'{clase}')

# Convertir las listas en matrices numpy
momentos_hu = np.array(momentos_hu)
perfiles = np.array(perfiles)
vectores=np.array(vectores)
# Guardar los resultados en archivos CSV
# df(momentos_hu, 'Momentos_hu')
# df(perfiles, 'Perfil')
# df(vectores, 'Vector')



#Función que segmenta, recorta y guarda las imágenes
def db_1():
    # Elección y lectura de imágenes desde local
    carpeta_i = r'D:\Upiita\6to\Patrones\Objetos\Placas base de datos'
    nombres = [f for f in os.listdir(carpeta_i) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Carpeta para guardar los recortes
    carpeta_r = os.path.join(carpeta_i, 'recortes')
    os.makedirs(carpeta_r, exist_ok=True)

    #Recorre cada imagen encontrada dentro de la carpeta
    for imagen in range(len(nombres)):
        ruta = os.path.join(carpeta_i, nombres[imagen])
        imagen_o = io.imread(ruta)

        binario=gbf(imagen_o)

        # Determinar Regiones
        etiquetas, num_objetos = measure.label(binario, return_num=True, connectivity=2)
        propiedades = measure.regionprops(etiquetas)

        # Crear una carpeta para los recortes de la imagen actual
        carpeta_ra = os.path.join(carpeta_r, f'recorte_{nombres[imagen]}')
        os.makedirs(carpeta_ra, exist_ok=True)

        # Objeto para la imagen segmentada
        plt.imshow(binario, cmap='gray')
        ax = plt.gca()

        # Recorrer las regiones detectadas
        for i, region in enumerate(propiedades):
            # Coordenadas del recuadro de cada objeto (mínimo y máximo)
            fila_min, columna_min, fila_max, columna_max = region.bbox

            # Dibujar un recuadro alrededor del objeto
            rect = plt.Rectangle((columna_min, fila_min), columna_max - columna_min, fila_max - fila_min,
                                edgecolor='red', facecolor='none', linewidth=2)
            #Agregar recuadros
            ax.add_patch(rect)

            # Extraer la región detectada
            recorte = binario[fila_min:fila_max, columna_min:columna_max]

            # Redimensionar la región para evitar discrepancias dimensionales
            recorte = resize(recorte, (40, 20), anti_aliasing=False)

            # Guardar el recorte
            ruta_r = os.path.join(carpeta_ra, f'region_{i}.png')
            io.imsave(ruta_r, recorte)

        # Guardar la imagen original segmentada por recuadros
        ruta_rs = os.path.join(carpeta_ra, f'Original_Segmentada_{nombres[imagen]}')
        plt.axis('off')
        plt.savefig(ruta_rs, bbox_inches='tight', pad_inches=0)
        plt.close()

# db_1()
