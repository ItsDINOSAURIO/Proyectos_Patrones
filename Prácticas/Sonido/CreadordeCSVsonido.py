import os
import pandas as pd
import numpy as np

# Función para procesar las imágenes de una carpeta de clases
def procesar_carpeta(clase, carpeta_clase):
    datos = []
    for ruta_actual,subcarpetas,archivos in os.walk(carpeta_clase):
        sonidos = [f for f in archivos if f.endswith(('.npy'))]
        
        for sonido_nombre in sonidos:
            ruta = os.path.join(ruta_actual, sonido_nombre)
            vector = np.load(ruta)

            datos.append([clase] + list(vector))
        
    return datos

# Función principal para leer las imágenes de cada clase y guardar en un CSV
def generar_base_datos(carpeta_principal, archivo_salida):
    clases = os.listdir(carpeta_principal)
    datos_vector = []
    
    # Procesar cada clase (carpeta)
    for clase in clases:
        carpeta_clase = os.path.join(carpeta_principal, clase)
        
        if os.path.isdir(carpeta_clase):  # Verificar que es una carpeta
            print(f"Procesando clase {clase}...")
            datos_clase = procesar_carpeta(clase, carpeta_clase)
            datos_vector.extend(datos_clase)
    
    # Crear un DataFrame con los resultados
    columnas = ['Clase'] + [f'Caracteristica {i}' for i in range(1, 151)]
    df = pd.DataFrame(datos_vector, columns=columnas)
    
    # Guardar en un archivo CSV
    df.to_csv(archivo_salida, index=False)
    print(f"Base de datos generada y guardada en {archivo_salida}")

# Ejemplo de uso
carpeta_principal = r"D:\Upiita\6to\Patrones\Practicas\Sonido"  # Carpeta donde están las subcarpetas de cada clase de número
archivo_salida = r"D:\Upiita\6to\Patrones\Practicas\Sonido\sonidos.csv"

# Generar la base de datos
generar_base_datos(carpeta_principal, archivo_salida)
