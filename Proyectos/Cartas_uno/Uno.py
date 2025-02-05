import matplotlib.pyplot as plt
from skimage import io, color
import numpy as np
import pandas as pd
from skimage.morphology import closing, disk
from skimage.transform import resize

# Función para calcular la correlación
def correlacion(recorte, clases_p):
    correlaciones = []
    for clase in clases_p:
        corr = np.corrcoef(recorte, clase)[0, 1] 
        correlaciones.append(corr)
    return np.array(correlaciones)

# Función para calcular el perfil de una imagen binarizada
def calcular_perfil(imagen_binaria, num_perfiles=10):
    perfil_valores = []
    num_filas = imagen_binaria.shape[0]

    # Dividir las filas en segmentos para obtener un número fijo de perfiles
    for i in range(num_perfiles):
        fila = int(i * (num_filas / num_perfiles))  # Escoge filas de manera uniforme
        if fila < num_filas:
            perfil_izquierdo = np.argmax(imagen_binaria[fila])  # Primer píxel blanco desde la izquierda
            perfil_derecho = imagen_binaria.shape[1] - 1 - np.argmax(np.flipud(imagen_binaria[fila]))  # Primer píxel blanco desde la derecha
            perfil_valores.append((perfil_izquierdo, perfil_derecho))
        else:
            perfil_valores.append((0, imagen_binaria.shape[1] - 1))  # No se encontró píxel blanco, retornar extremos
    
    # Aplanar la lista de perfiles para un formato más fácil de manejar
    return [item for sublist in perfil_valores for item in sublist]

# Función principal para procesar la imagen y encontrar la correlación con las clases del CSV
def procesar_imagen_y_correlacionar(imagen_path, csv_path):
    # Leer la imagen
    imagen = io.imread(imagen_path)
    imagen = resize(imagen, (700, 400), anti_aliasing=False)
    gris = color.rgb2gray(imagen)

    # Binarización y procesamiento morfológico
    binario = gris > 0.98
    recorte = binario[6:131, 0:125]
    estructura = disk(1)
    ero = closing(recorte, estructura)

    # Calcular el perfil de la imagen procesada
    perfil1 = calcular_perfil(ero)

    # Leer el archivo CSV con las clases y los perfiles de referencia
    datos_clases = pd.read_csv(csv_path)
    clases = datos_clases['Clase'].values
    perfiles_clases = datos_clases.drop(columns=['Clase']).values

    # Calcular la correlación con cada perfil de la clase
    correlaciones_perf = correlacion(perfil1, perfiles_clases)
    imax_corr_perf = np.argmax(correlaciones_perf)
    max_corr_perf = correlaciones_perf[imax_corr_perf]

    # Asignar la clase con mayor correlación
    if max_corr_perf > 0.99:
        resultado_correlacion = clases[imax_corr_perf]
    else:
        resultado_correlacion = 'Desconocida'
    

    print(resultado_correlacion)
    plt.figure(1)
    plt.imshow(binario,cmap='gray')    
    plt.grid(False)
    plt.show()
    return resultado_correlacion

# Uso de la función con la ruta de la imagen y el archivo CSV
imagen_path = r'C:\Users\erikn\Documents\School\7mo semestre\Reconocimiento de patrones\Cartas_uno\Colores\3_rojo.jpg'
csv_path = r'C:\Users\erikn\Documents\School\7mo semestre\Reconocimiento de patrones\Cartas_uno\archivo_salida.csv'
resultado = procesar_imagen_y_correlacionar(imagen_path, csv_path)
print(f'La clase asignada a la imagen es: {resultado}')
