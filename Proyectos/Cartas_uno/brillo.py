import os
from PIL import Image, ImageEnhance, ImageStat

def calcular_brillo(imagen):
    # Convertir la imagen a escala de grises
    imagen_gris = imagen.convert('L')
    
    # Obtener el brillo promedio
    stat = ImageStat.Stat(imagen_gris)
    brillo_promedio = stat.mean[0]
    
    return brillo_promedio

def ajustar_brillo(imagen, brillo_actual, brillo_deseado):
    # Calcular el factor de ajuste necesario
    factor_ajuste = brillo_deseado / brillo_actual

    # Ajustar el brillo de la imagen
    enhancer = ImageEnhance.Brightness(imagen)
    imagen_brillo_ajustado = enhancer.enhance(factor_ajuste)

    return imagen_brillo_ajustado

def igualar_brillo_carpeta(carpeta_entrada, carpeta_salida, brillo_deseado):
    # Crear la carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    # Recorrer todos los archivos en la carpeta de entrada
    for archivo in os.listdir(carpeta_entrada):
        # Verificar si el archivo es una imagen
        if archivo.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Cargar la imagen
            ruta_imagen = os.path.join(carpeta_entrada, archivo)
            imagen = Image.open(ruta_imagen)

            # Calcular el brillo actual de la imagen
            brillo_actual = calcular_brillo(imagen)
            print(f"Brillo actual de {archivo}: {brillo_actual}")

            # Ajustar la imagen al brillo deseado
            imagen_ajustada = ajustar_brillo(imagen, brillo_actual, brillo_deseado)

            # Guardar la imagen modificada en la carpeta de salida
            ruta_salida = os.path.join(carpeta_salida, archivo)
            imagen_ajustada.save(ruta_salida)
            print(f"Brillo ajustado y guardado: {ruta_salida}")

# Ejemplo de uso
# Ejemplo de uso
carpeta_entrada = r'C:\Users\erikn\Documents\School\7mo semestre\Reconocimiento de patrones\Cartas_uno\Azul'
carpeta_salida = r'C:\Users\erikn\Documents\School\7mo semestre\Reconocimiento de patrones\Cartas_uno\Azul'
brillo_deseado = 200  # Brillo deseado (valor entre 0 y 255)

igualar_brillo_carpeta(carpeta_entrada, carpeta_salida, brillo_deseado)

