import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

# PERSONAS MORENAS

# Cargar la imagen que contiene diferentes tonos de piel
imagen = cv2.imread(r"D:\Upiita\6to\Patrones\Objetos\Piel\img4.jpg")
imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# Mostrar la imagen original
plt.figure(0)
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title("Imagen Original")
plt.axis('off')

# Coordenadas (y0:y1, x0:x1) para muestras de piel seleccionadas manualmente
muestra1 = imagen_hsv[847:901, 601:717] 
muestra2 = imagen_hsv[291:275, 394:499]
muestra3 = imagen_hsv[367:415, 360:404]

# Concatenar las muestras para procesarlas en conjunto
muestras = np.vstack((muestra1.reshape(-1, 3), muestra2.reshape(-1, 3), muestra3.reshape(-1, 3)))

# Extraer las componentes H y S de las muestras
H = muestras[:, 0]  # Canal H (Tono)
S = muestras[:, 1]  # Canal S (Saturación)

# Calcular media y desviación estándar para H y S
media_H, std_H = np.mean(H), np.std(H)
media_S, std_S = np.mean(S), np.std(S)

print(f"Media H: {media_H}, Desviación estándar H: {std_H}")
print(f"Media S: {media_S}, Desviación estándar S: {std_S}")

def probabilidad_normal(x, media, desviacion):
    # Implementación de la función de densidad de una distribución normal
    return (1 / (np.sqrt(2 * np.pi) * desviacion)) * np.exp(-0.5 * ((x - media) / desviacion) ** 2)

def es_piel(h, s, media_H, std_H, media_S, std_S):
    # Calcular las probabilidades para H y S
    p_h = probabilidad_normal(h, media_H, std_H)
    p_s = probabilidad_normal(s, media_S, std_S)
    # Usar Naive Bayes para calcular la probabilidad conjunta
    return p_h * p_s

# Crear una máscara vacía para la piel
mascara_piel = np.zeros((imagen_hsv.shape[0], imagen_hsv.shape[1]), dtype=np.uint8)

# Iterar sobre cada píxel y aplicar el clasificador
for y in range(imagen_hsv.shape[0]):
    for x in range(imagen_hsv.shape[1]):
        h, s, _ = imagen_hsv[y, x]  # Extraer valores H y S del píxel
        if es_piel(h, s, media_H, std_H, media_S, std_S) > 1e-6:  # Umbral ajustable
            mascara_piel[y, x] = 255  # Marcar como piel

# Mostrar la máscara de la piel
plt.figure(1)
plt.imshow(mascara_piel, cmap='gray')
plt.title("Máscara de Piel")
plt.axis('off')

# ------------------------------------------------------------------------------------

# Guardar los valores en un archivo pickle
with open("valores_piel.pkl", "wb") as f:
    pickle.dump((media_H, std_H, media_S, std_S), f)

print("Valores guardados en 'valores_piel.pkl'")

# Cargar los valores guardados
with open("valores_piel.pkl", "rb") as f:
    media_H, std_H, media_S, std_S = pickle.load(f)

print(f"Media H: {media_H}, Desviación estándar H: {std_H}")
print(f"Media S: {media_S}, Desviación estándar S: {std_S}")

# Función para clasificar si un píxel es piel
def probabilidad_normal(x, media, desviacion):
    return (1 / (np.sqrt(2 * np.pi) * desviacion)) * np.exp(-0.5 * ((x - media) / desviacion) ** 2)

def es_piel(h, s, media_H, std_H, media_S, std_S):
    p_h = probabilidad_normal(h, media_H, std_H)
    p_s = probabilidad_normal(s, media_S, std_S)
    return p_h * p_s

# Cargar una nueva imagen
imagen_nueva = cv2.imread(r"D:\Upiita\6to\Patrones\Objetos\Piel\img5.jpg")
imagen_nueva_hsv = cv2.cvtColor(imagen_nueva, cv2.COLOR_BGR2HSV)

# Crear una máscara para detectar la piel
mascara_piel = np.zeros((imagen_nueva_hsv.shape[0], imagen_nueva_hsv.shape[1]), dtype=np.uint8)

# Aplicar el filtro en cada píxel
for y in range(imagen_nueva_hsv.shape[0]):
    for x in range(imagen_nueva_hsv.shape[1]):
        h, s, _ = imagen_nueva_hsv[y, x]
        if es_piel(h, s, media_H, std_H, media_S, std_S) > 1e-6:  # Umbral ajustable
            mascara_piel[y, x] = 255

# Mostrar la máscara
plt.figure(2)
plt.imshow(mascara_piel, cmap='gray')
plt.title("Máscara de Piel - Imagen Nueva")
plt.axis('off')
plt.show()

# import pickle
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # PERSONAS NO TAN MORENAS

# # Cargar la imagen que contiene diferentes tonos de piel
# imagen = cv2.imread(r"D:\Upiita\6to\Patrones\Objetos\Piel\img1.jpg")
# imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# # Mostrar la imagen original
# plt.figure(0)
# plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
# plt.title("Imagen Original")
# plt.axis('off')

# # Coordenadas (y0:y1, x0:x1) para muestras de piel seleccionadas manualmente
# muestra1 = imagen_hsv[268:210, 526:608] 
# muestra2 = imagen_hsv[415:486, 611:666]
# muestra3 = imagen_hsv[632:683, 496:591]

# # Concatenar las muestras para procesarlas en conjunto
# muestras = np.vstack((muestra1.reshape(-1, 3), muestra2.reshape(-1, 3), muestra3.reshape(-1, 3)))

# # Extraer las componentes H y S de las muestras
# H = muestras[:, 0]  # Canal H (Tono)
# S = muestras[:, 1]  # Canal S (Saturación)

# # Calcular media y desviación estándar para H y S
# media_H, std_H = np.mean(H), np.std(H)
# media_S, std_S = np.mean(S), np.std(S)

# print(f"Media H: {media_H}, Desviación estándar H: {std_H}")
# print(f"Media S: {media_S}, Desviación estándar S: {std_S}")

# def probabilidad_normal(x, media, desviacion):
#     # Implementación de la función de densidad de una distribución normal
#     return (1 / (np.sqrt(2 * np.pi) * desviacion)) * np.exp(-0.5 * ((x - media) / desviacion) ** 2)

# def es_piel(h, s, media_H, std_H, media_S, std_S):
#     # Calcular las probabilidades para H y S
#     p_h = probabilidad_normal(h, media_H, std_H)
#     p_s = probabilidad_normal(s, media_S, std_S)
#     # Usar Naive Bayes para calcular la probabilidad conjunta
#     return p_h * p_s

# # Crear una máscara vacía para la piel
# mascara_piel = np.zeros((imagen_hsv.shape[0], imagen_hsv.shape[1]), dtype=np.uint8)

# # Iterar sobre cada píxel y aplicar el clasificador
# for y in range(imagen_hsv.shape[0]):
#     for x in range(imagen_hsv.shape[1]):
#         h, s, _ = imagen_hsv[y, x]  # Extraer valores H y S del píxel
#         if es_piel(h, s, media_H, std_H, media_S, std_S) > 1e-6:  # Umbral ajustable
#             mascara_piel[y, x] = 255  # Marcar como piel

# # Mostrar la máscara de la piel
# plt.figure(1)
# plt.imshow(mascara_piel, cmap='gray')
# plt.title("Máscara de Piel")
# plt.axis('off')

# # ------------------------------------------------------------------------------------

# # Guardar los valores en un archivo pickle
# with open("valores_piel.pkl", "wb") as f:
#     pickle.dump((media_H, std_H, media_S, std_S), f)

# print("Valores guardados en 'valores_piel.pkl'")

# # Cargar los valores guardados
# with open("valores_piel.pkl", "rb") as f:
#     media_H, std_H, media_S, std_S = pickle.load(f)

# print(f"Media H: {media_H}, Desviación estándar H: {std_H}")
# print(f"Media S: {media_S}, Desviación estándar S: {std_S}")

# # Función para clasificar si un píxel es piel
# def probabilidad_normal(x, media, desviacion):
#     return (1 / (np.sqrt(2 * np.pi) * desviacion)) * np.exp(-0.5 * ((x - media) / desviacion) ** 2)

# def es_piel(h, s, media_H, std_H, media_S, std_S):
#     p_h = probabilidad_normal(h, media_H, std_H)
#     p_s = probabilidad_normal(s, media_S, std_S)
#     return p_h * p_s

# # Cargar una nueva imagen
# imagen_nueva = cv2.imread(r"D:\Upiita\6to\Patrones\Objetos\Piel\img3.jpg")
# imagen_nueva_hsv = cv2.cvtColor(imagen_nueva, cv2.COLOR_BGR2HSV)

# # Crear una máscara para detectar la piel
# mascara_piel = np.zeros((imagen_nueva_hsv.shape[0], imagen_nueva_hsv.shape[1]), dtype=np.uint8)

# # Aplicar el filtro en cada píxel
# for y in range(imagen_nueva_hsv.shape[0]):
#     for x in range(imagen_nueva_hsv.shape[1]):
#         h, s, _ = imagen_nueva_hsv[y, x]
#         if es_piel(h, s, media_H, std_H, media_S, std_S) > 1e-6:  # Umbral ajustable
#             mascara_piel[y, x] = 255

# # Mostrar la máscara
# plt.figure(2)
# plt.imshow(mascara_piel, cmap='gray')
# plt.title("Máscara de Piel - Imagen Nueva")
# plt.axis('off')
# plt.show()

# import pickle
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # PERSONAS BLANCAS

# # Cargar la imagen que contiene diferentes tonos de piel
# imagen = cv2.imread(r"D:\Upiita\6to\Patrones\Objetos\Piel\img9.jpg")
# imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# # Mostrar la imagen original
# plt.figure(0)
# plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
# plt.title("Imagen Original")
# plt.axis('off')

# # Coordenadas (y0:y1, x0:x1) para muestras de piel seleccionadas manualmente
# muestra1 = imagen_hsv[306:350, 601:635] 
# muestra2 = imagen_hsv[469:489, 581:635]
# muestra3 = imagen_hsv[1235:1272, 332:383]

# # Concatenar las muestras para procesarlas en conjunto
# muestras = np.vstack((muestra1.reshape(-1, 3), muestra2.reshape(-1, 3), muestra3.reshape(-1, 3)))

# # Extraer las componentes H y S de las muestras
# H = muestras[:, 0]  # Canal H (Tono)
# S = muestras[:, 1]  # Canal S (Saturación)

# # Calcular media y desviación estándar para H y S
# media_H, std_H = np.mean(H), np.std(H)
# media_S, std_S = np.mean(S), np.std(S)

# print(f"Media H: {media_H}, Desviación estándar H: {std_H}")
# print(f"Media S: {media_S}, Desviación estándar S: {std_S}")

# def probabilidad_normal(x, media, desviacion):
#     # Implementación de la función de densidad de una distribución normal
#     return (1 / (np.sqrt(2 * np.pi) * desviacion)) * np.exp(-0.5 * ((x - media) / desviacion) ** 2)

# def es_piel(h, s, media_H, std_H, media_S, std_S):
#     # Calcular las probabilidades para H y S
#     p_h = probabilidad_normal(h, media_H, std_H)
#     p_s = probabilidad_normal(s, media_S, std_S)
#     # Usar Naive Bayes para calcular la probabilidad conjunta
#     return p_h * p_s

# # Crear una máscara vacía para la piel
# mascara_piel = np.zeros((imagen_hsv.shape[0], imagen_hsv.shape[1]), dtype=np.uint8)

# # Iterar sobre cada píxel y aplicar el clasificador
# for y in range(imagen_hsv.shape[0]):
#     for x in range(imagen_hsv.shape[1]):
#         h, s, _ = imagen_hsv[y, x]  # Extraer valores H y S del píxel
#         if es_piel(h, s, media_H, std_H, media_S, std_S) > 1e-6:  # Umbral ajustable
#             mascara_piel[y, x] = 255  # Marcar como piel

# # Mostrar la máscara de la piel
# plt.figure(1)
# plt.imshow(mascara_piel, cmap='gray')
# plt.title("Máscara de Piel")
# plt.axis('off')

# # ------------------------------------------------------------------------------------

# # Guardar los valores en un archivo pickle
# with open("valores_piel.pkl", "wb") as f:
#     pickle.dump((media_H, std_H, media_S, std_S), f)

# print("Valores guardados en 'valores_piel.pkl'")

# # Cargar los valores guardados
# with open("valores_piel.pkl", "rb") as f:
#     media_H, std_H, media_S, std_S = pickle.load(f)

# print(f"Media H: {media_H}, Desviación estándar H: {std_H}")
# print(f"Media S: {media_S}, Desviación estándar S: {std_S}")

# # Función para clasificar si un píxel es piel
# def probabilidad_normal(x, media, desviacion):
#     return (1 / (np.sqrt(2 * np.pi) * desviacion)) * np.exp(-0.5 * ((x - media) / desviacion) ** 2)

# def es_piel(h, s, media_H, std_H, media_S, std_S):
#     p_h = probabilidad_normal(h, media_H, std_H)
#     p_s = probabilidad_normal(s, media_S, std_S)
#     return p_h * p_s

# # Cargar una nueva imagen
# imagen_nueva = cv2.imread(r"D:\Upiita\6to\Patrones\Objetos\Piel\img12.jpg")       
# imagen_nueva_hsv = cv2.cvtColor(imagen_nueva, cv2.COLOR_BGR2HSV)

# # Crear una máscara para detectar la piel
# mascara_piel = np.zeros((imagen_nueva_hsv.shape[0], imagen_nueva_hsv.shape[1]), dtype=np.uint8)

# # Aplicar el filtro en cada píxel
# for y in range(imagen_nueva_hsv.shape[0]):
#     for x in range(imagen_nueva_hsv.shape[1]):
#         h, s, _ = imagen_nueva_hsv[y, x]
#         if es_piel(h, s, media_H, std_H, media_S, std_S) > 1e-6:  # Umbral ajustable
#             mascara_piel[y, x] = 255

# # Mostrar la máscara
# plt.figure(2)
# plt.imshow(mascara_piel, cmap='gray')
# plt.title("Máscara de Piel - Imagen Nueva")
# plt.axis('off')
# plt.show()