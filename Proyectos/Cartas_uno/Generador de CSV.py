import os
import pandas as pd
from skimage import io, color
from skimage.morphology import closing, disk
from skimage.transform import resize
import numpy as np

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
    if len(sil) < 80:
        sil.extend([0] * (80 - len(sil)))
    elif len(sil) > 80:
        sil = sil[:80]
    return sil

# Función para procesar las imágenes de una carpeta de clases
def procesar_carpeta(clase, carpeta_clase):
    perfiles = []
    imagenes = [f for f in os.listdir(carpeta_clase) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for imagen_nombre in imagenes:
        ruta_imagen = os.path.join(carpeta_clase, imagen_nombre)
        imagen = io.imread(ruta_imagen)  # Cargar la imagen
        
        # Preprocesar la imagen
        imagen_resized = resize(imagen, (700, 400), anti_aliasing=False)  # Redimensionar la imagen
            # Verifica que la imagen se está procesando correctamente
        print(f"Procesando imagen: {imagen_nombre}, tamaño: {imagen_resized.shape}")
    
        imagen_gris = color.rgb2gray(imagen_resized)  # Convertir a escala de grises
        imagen_binaria = imagen_gris > 0.98  # Binarizar la imagen
        
        # Recortar y aplicar una operación morfológica
        recorte = imagen_binaria[6:160, 0:160]
        estructura = disk(1)
        imagen_procesada = closing(recorte, estructura)
        
        # Calcular el perfil
        perfil_valores = perfil(imagen_procesada)
        
        # Añadir el perfil a la lista, junto con la clase
        perfiles.append([clase] + perfil_valores)
        print(f"Número de características: {len(perfil_valores)}")
    
    return perfiles

# Función principal para leer las imágenes de cada clase y guardar en un CSV
def generar_base_datos(carpeta_principal, archivo_salida):
    clases = os.listdir(carpeta_principal)
    datos_perfiles = []
    
    max_caracteristicas = 0  # Variable para almacenar la cantidad máxima de características

    
    # Procesar cada clase (carpeta)
    for clase in clases:
        carpeta_clase = os.path.join(carpeta_principal, clase)
        
        if os.path.isdir(carpeta_clase):  # Verificar que es una carpeta
            print(f"Procesando clase {clase}...")
            perfiles_clase = procesar_carpeta(clase, carpeta_clase)
            if perfiles_clase:  # Verificar que hay datos
                datos_perfiles.extend(perfiles_clase)
                # Actualizar el número máximo de características
                for perfil in perfiles_clase:
                    if len(perfil) > max_caracteristicas:
                        max_caracteristicas = len(perfil)
            else:
                print(f"No se encontraron perfiles en la clase {clase}.")
    
    # Verificar si hay datos para generar el DataFrame
    if datos_perfiles:
        
        columnas = ['Clase'] + [f'Caracteristica {i}' for i in range(1, max_caracteristicas)]
        datos_perfiles = [perfil + [np.nan] * (max_caracteristicas - len(perfil)) for perfil in datos_perfiles]
        df = pd.DataFrame(datos_perfiles, columns=columnas)
        
        # Guardar en un archivo CSV
        df.to_csv(archivo_salida, index=False)
        print(f"Base de datos generada y guardada en {archivo_salida}")
    else:
        print("No se generaron datos de perfiles, no se creó ningún archivo CSV.")

# Ejemplo de uso
carpeta_principal = r'C:\Users\erikn\Documents\School\7mo semestre\Reconocimiento de patrones\Cartas_uno\Comodin'  # Reemplaza con la ruta a tu carpeta principal que contiene las subcarpetas de las clases
archivo_salida = r'C:\Users\erikn\Documents\School\7mo semestre\Reconocimiento de patrones\Cartas_uno\comodin.csv'  # Reemplaza con la ruta donde quieras guardar el CSV

# Generar la base de datos
generar_base_datos(carpeta_principal, archivo_salida)

