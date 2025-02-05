import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Ruta de la carpeta donde están almacenadas las imágenes
carpeta_imagenes = r"C:\Users\erikn\Documents\School\7mo semestre\Reconocimiento de patrones\Cartas_uno\Colores"

# Solicitar el número y el color
numero = input("Introduce el número de la carta: ").lower()  # Convertir a minúsculas
color = input("Introduce el color de la carta: ").lower()  # Convertir a minúsculas

# Formar el nombre del archivo
nombre_imagen = f"{numero}_{color}.jpg"

# Construir la ruta completa de la imagen
ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)

# Verificar si la imagen existe
if os.path.exists(ruta_imagen):
    # Cargar y mostrar la imagen
    img = mpimg.imread(ruta_imagen)
    plt.imshow(img)
    plt.axis('off')  # Opcional: Ocultar los ejes
    plt.show()
else:
    print(f"La imagen '{nombre_imagen}' no fue encontrada en la carpeta {carpeta_imagenes}.")
