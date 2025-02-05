import os
from skimage import io, color
from skimage.transform import resize
from skimage.morphology import closing, disk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageStat
import random
import matplotlib.image as mpimg
import cv2

csv_path = r"D:\Upiita\6to\Patrones\Cartas_uno\Cartas_uno\perfiles.csv"


def capturar_imagen_webcam():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return None

    print("Presiona 'Espacio' para capturar la imagen, 'Esc' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar la imagen.")
            break

        cv2.imshow('Captura', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            print("Imagen capturada.")
            cap.release()
            cv2.destroyAllWindows()
           
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elif key == 27:  
            print("Saliendo sin capturar.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return None  # Si no se captura una imagen

def calcular_brillo(imagen):
    # Convertir imagen PIL a numpy array si no es ya un array
    if isinstance(imagen, Image.Image):
        imagen = np.array(imagen)

    if imagen.dtype != np.uint8:
        imagen = imagen.astype(np.uint8) 
        # raise ValueError("El tipo de dato de la imagen debe ser uint8.")

    # Calcular el brillo como el promedio de los valores de los píxeles
    return np.mean(imagen)

def ajustar_brillo(imagen, brillo_actual, brillo_deseado):
    if imagen.dtype != np.uint8:
        imagen = (imagen * 255).astype(np.uint8)
    factor_ajuste = brillo_deseado / brillo_actual
    imagen_pil = Image.fromarray(imagen)
    enhancer = ImageEnhance.Brightness(imagen_pil)
    imagen_brillo_ajustado = enhancer.enhance(factor_ajuste)
    imagen_ajustada = np.array(imagen_brillo_ajustado)
    return imagen_brillo_ajustado

def igualar_brillo(brillo_deseado, imagen):
    brillo_actual = calcular_brillo(imagen)  # Llamada corregida
    factor = brillo_deseado / brillo_actual
    imagen_np = np.array(imagen)  # Convertimos la imagen a ndarray
    imagen_np = np.clip(imagen_np * factor, 0, 255).astype(np.uint8)  # Ajustamos el brillo
    return imagen_np  # Convertimos de nuevo a PIL

def correlacion(recorte, clases_p):
    correlaciones = []
    for clase in clases_p:
        corr = np.corrcoef(recorte, clase)[0, 1] 
        correlaciones.append(corr)
    return np.array(correlaciones)

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
    

def distancia_euclidiana(punto, clase):
    punto = np.array(punto)
    clase = np.array(clase)
    return np.linalg.norm(punto - clase)

def detector(imagen,path):
    brillo_deseado = 200  
    # imagen = io.imread(ruta_imagen)
    imagen = imagen[ 100:702,130:488,:]

    plt.figure(0)
    plt.imshow(imagen)
    plt.title('Imagen Original')
    plt.axis('off')
    plt.show()

    imagen = igualar_brillo(brillo_deseado,imagen)
    plt.figure(0)
    plt.imshow(imagen)
    plt.title('Imagen con aumento de brillo')
    plt.axis('off')
    plt.show()   

        # Convertir la imagen a formato numpy.ndarray si es necesario
    if not isinstance(imagen, np.ndarray):
        imagen = np.array(imagen)
    
    imagen = resize(imagen, (700, 400), anti_aliasing=False)

    if imagen.max() <= 1:
        imagen1 = (imagen * 255).astype(np.uint8)
                
    imagen1 = imagen1[150:170,0:30,:]
    plt.figure(0)
    plt.imshow(imagen1)
    plt.title('Imagen recortada para detectar color')
    plt.axis('off')
    plt.show()       

    colores_referencia = {
        'Rojo': [255, 0, 0],
        'Verde': [186, 236, 130],
        'Azul': [142, 205, 251],
        'Amarillo': [255, 255, 0],
        'Negro': [0, 0, 0]
    }

    pixels = imagen1.reshape(-1, 3)

    def distancia_mahalanobis_manual(color, referencia, cov_inv):
        delta = color - referencia
        return np.sqrt(np.dot(np.dot(delta.T, cov_inv), delta))

    valores_referencia = np.array(list(colores_referencia.values()))
    cov_matrix = np.cov(valores_referencia, rowvar=False)
    cov_inv = np.linalg.inv(cov_matrix)

    conteo_colores = {nombre: 0 for nombre in colores_referencia.keys()}
    rgb_detectados = []
    for pixel in pixels:
        distancias = []
        for nombre, referencia in colores_referencia.items():
            dist = distancia_mahalanobis_manual(pixel, referencia, cov_inv)
            distancias.append((dist, nombre))
        color_cercano = min(distancias, key=lambda x: x[0])[1]
        conteo_colores[color_cercano] += 1
        rgb_detectados.append(pixel)
    rgb_detectados = np.array(rgb_detectados)
    color_predominante = max(conteo_colores, key=conteo_colores.get)
               
    if color_predominante == 'Negro':
        comodin = r"D:\Upiita\6to\Patrones\Cartas_uno\Cartas_uno\comodin.csv"
        resultado_correlacion = correlacionar(imagen,comodin)
        if resultado_correlacion == 'Desconocida':
            resultado_correlacion = 'Cambio'
    else:
        resultado_correlacion = correlacionar(imagen,path)
        if color_predominante == 'Rojo':
            if resultado_correlacion == 'Desconocida':
                resultado_correlacion = '+2'

    return color_predominante,resultado_correlacion

def correlacionar(imagen, csv_path):
    # imagen = io.imread(imagen_path)
    if isinstance(imagen, Image.Image):
        imagen = np.array(imagen)

    brillo_deseado = 200  
    imagen = igualar_brillo(brillo_deseado,imagen)
    
    imagen = resize(imagen, (700, 400), anti_aliasing=False)
    gris = color.rgb2gray(imagen)
    plt.figure(0)
    plt.imshow(gris,cmap = 'gray')
    plt.title('Imagen en grises')
    plt.axis('off')
    plt.show()   

    binario = gris > 0.999
    plt.imshow(binario,cmap = 'gray')
    plt.title('Imagen binarizada')
    plt.axis('off')
    plt.show()     
    recorte = binario[0:133, 0:125]
    plt.imshow(recorte,cmap = 'gray')
    plt.title('Imagen con aumento de brillo')
    plt.axis('off')
    plt.show()     

    estructura = disk(1)
    ero = closing(recorte, estructura)

    perfil1 = perfil(ero)

    datos_clases = pd.read_csv(csv_path)
    clases = datos_clases['Clase'].values
    perfiles_clases = datos_clases.drop(columns=['Clase']).values

    correlaciones_perf = correlacion(perfil1, perfiles_clases)
    imax_corr_perf = np.argmax(correlaciones_perf)
    max_corr_perf = correlaciones_perf[imax_corr_perf]
    print(max_corr_perf)

    # Asignar la clase con mayor correlación
    if max_corr_perf > 0.6:
        resultado_correlacion = clases[imax_corr_perf]
    else:
        resultado_correlacion = 'Desconocida'
    return resultado_correlacion
resultados =[]
color_cartas = []
digito = []
for i in range(3):
    print(f"Capturando imagen {i + 1}...")
    imagen = capturar_imagen_webcam()
    nombre_archivo = f"{i + 1}.png"
    ruta_completa = os.path.join(r"D:\Upiita\6to\Patrones\Cartas_uno\Cartas_uno\Mazo\Nuevo", nombre_archivo)
    imagen.save(ruta_completa)
    if isinstance(imagen, Image.Image):
        imagen = np.array(imagen)
    imagen=imagen[:,387:900]
    plt.imshow(imagen)
    plt.show()
    if imagen is not None:
        color_predominante,resultado_correlacion = detector(imagen, csv_path)
        color_cartas.append(color_predominante)
        digito.append(resultado_correlacion)
        resultados.append([color_predominante, resultado_correlacion])
        print(f"Imagen {i + 1}: Color predominante - {color_predominante}, Clase detectada - {resultado_correlacion}")
    else:
        print(f"Imagen {i + 1}: No se pudo capturar.")
# for i, (color, clase) in enumerate(resultados):
#     print(f"Imagen {i + 1}: Color predominante - {color}, Clase detectada - {clase}")
# for i in range(8):
#     ruta_imagen = input(f"Ingrese la ruta de la imagen {i + 1} : ")
#     color_predominante, resultado_correlacion = detector(ruta_imagen,csv_path)
#     color_cartas.append(color_predominante)
#     digito.append(resultado_correlacion)
#     resultados.append([color_predominante, resultado_correlacion])


print("Simulación de juego de UNO")


while True:
    carpeta_imagenes = r"D:\Upiita\6to\Patrones\Cartas_uno\Cartas_uno\Colores"
    
    colores_random = []
    print(resultados)
    
    cartas_humano = input("Ingrese el numero de cartas que tienes: ")

    if cartas_humano == 0:
        print('Haz ganado tu')
        break

    for i in range(len(resultados)):
        colores = resultados[i][0]
        colores_random.append(colores)
        
    if len(resultados) == 0:
        print("He ganado el juego.")
        break
    
    if len(resultados) == 1:
        print("Uno")   
    print(' ')
    
    print('¿Qué carta se encuentra en la mesa de juego?')
    print(' ')
    numero = input("Ingrese el número o comodín de la carta: ")
    color = input("Ingrese el color de la carta: ")
    print(' ')

    carta_jugada = False 

    if str(numero) == '+2':
        for _ in range(2):
            print('Ingresa las +2 cartas que tocaron:')
            numero1 = input("Ingrese el número o comodín de la carta: ")
            color1 = input("Ingrese el color de la carta: ")
            carta = [str(color1), str(numero1)]
            resultados.append(carta)
            carta_jugada = True

    elif str(numero) == '+4':
        for _ in range(4):
            print('Ingresa las +4 cartas que tocaron')
            numero1 = input("Ingrese el número o comodín de la carta: ")
            color1 = input("Ingrese el color de la carta: ")
            carta = [str(color1), str(numero1)]
            resultados.append(carta)
        color1 = input("Ingrese el color que debo tirar: ")
        for i in range(len(resultados)):
            if resultados[i][0] == color1:
                print(' ')
                nombre_imagen = f"{numero1.lower()}_{color1.lower()}.jpg"
                ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)
                print(f"Tiro la siguiente carta: {resultados[i][0]}, {resultados[i][1]}")
                del resultados[i]  # Eliminar la carta jugada
                carta_jugada = True
                img = mpimg.imread(ruta_imagen)
                plt.imshow(img)
                plt.axis('off')  # Opcional: Ocultar los ejes
                plt.show()
                break
                
    elif str(numero) == 'cambio':
        color1 = input("Ingrese el color que debo tirar: ")
        for i in range(len(resultados)):
            if resultados[i][0] == color1:
                print(' ')
                print(f"Tiro la siguiente carta: {resultados[i][0]}, {resultados[i][1]}")
                del resultados[i]  # Eliminar la carta jugada
                carta_jugada = True
                break
                
    elif str(numero) == 'salto':
        print('Me quitaron mi turno :((')
        carta_jugada = True
        
    elif str(numero) == 'reversa':
        print('Se cambia el sentido del turno')
        carta_jugada = True     
      
    if not carta_jugada:
        for i in range(len(resultados)):
            if resultados[i][0] == color or resultados[i][1] == numero:
                nombre_imagen = f"{resultados[i][1].lower()}_{resultados[i][0].lower()}.jpg"
                ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)
                print(f"Tiro la siguiente carta: {resultados[i][0]}, {resultados[i][1]}")
                del resultados[i]  # Eliminar la carta jugada
                carta_jugada=True
                
                img = mpimg.imread(ruta_imagen)
                plt.imshow(img)
                plt.axis('off')  # Opcional: Ocultar los ejes
                plt.show()
                break

            elif resultados[i][0] == 'Negro':
                print(f'Tiro el comodin {resultados[i][1]}')
                nombre_imagen = f"{resultados[i][1]}.jpg"
                ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)
                print(f"Escojo el color: {random.choice(colores_random)}")
                del resultados[i]  # Eliminar la carta jugada
                carta_jugada = True
                img = mpimg.imread(ruta_imagen)
                plt.imshow(img)
                plt.axis('off')  # Opcional: Ocultar los ejes
                plt.show()
                break
                

    if not carta_jugada:
        # Si no hay cartas para tirar, pedir más cartas
        while True:
            print('No poseo ninguna carta que tirar, ingresa cartas para ampliar el mazo.')
            numero1 = input("Ingrese el número o comodín de la carta: ")
            color1 = input("Ingrese el color de la carta: ")
            carta = [str(color1), str(numero1)]
            resultados.append(carta)

            # Verificar si la nueva carta es jugable
            if str(numero1) == numero or str(color1) == color:
                nombre_imagen = f"{numero1.lower()}_{color1.lower()}.jpg"
                ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)
                print(f"Tiro la siguiente carta: {color1}, {numero1}")
                resultados.remove(carta)  # Jugar la carta
                break

