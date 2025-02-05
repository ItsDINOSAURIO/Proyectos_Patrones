from PIL import Image
from numpy import array, zeros_like
from pylab import figure, imshow, block
from scipy.ndimage import label, generate_binary_structure, binary_fill_holes
from matplotlib.patches import Rectangle
from drawnow import drawnow
import matplotlib.pyplot as mpl
import time
import numpy as np
 
def repite():
    mpl.imshow(block([[A], [B]]))
 
# ======================= Primera Parte ================================== #
print('Primera parte del programa: Entrenamiento del sistema')
 
# 1. Cargar imágenes de entrenamiento
bin1 = array(Image.open(r'D:\Upiita\6to\Patrones\Objetos\entrena6.bmp'))
bin2 = array(Image.open(r'D:\Upiita\6to\Patrones\Objetos\entrena9.bmp'))
 
# 2. Mostrar las imágenes de entrenamiento
figure(1)
imshow(bin1)
figure(2)
imshow(bin2)
 
# 3. Generar estructura binaria y etiquetar las imágenes
con8 = generate_binary_structure(2, 2)
Ima6, Can6 = label(bin1, structure=con8)
Ima9, Can9 = label(bin2, structure=con8)
figure(3)
imshow(Ima6)
figure(4)
imshow(Ima9)
 
# 4. Procesar cada imagen etiquetada
for j in range(1, 3):
    if j == 1:
        numero, cantidad = Ima6, Can6
        print('Se procesa el número 6')
    else:
        numero, cantidad = Ima9, Can9
        print('Se procesa el número 9')
 
    dato = []
    # 5. Procesar objetos en la imagen
    for i in range(1, cantidad):
        print(f'Objeto {i} de {cantidad}')
        s1 = np.where(numero == i, 1, 0) #Segmenta la imagen
        s2 = binary_fill_holes(s1).astype(int) #Rellena la figura
        s3 = np.logical_xor(s1, s2) #Segmenta el relleno con una XOR entre ambas imágenes
        Fil1, Col1 = np.nonzero(s1)
        Fil2, Col2 = np.nonzero(s3)
        objmin, objmax = min(Fil2), max(Fil2)
        NFmin, NFmax = min(Fil1), max(Fil1)
        medio = (objmax - objmin) / 2.0 + objmin#Se posiciona en la mitad de la imagen, así determina si el relleno se encuentra abajo o arriba del objeto para diferenciar entre 6 y 9
        dato.append((medio - NFmin) / (NFmax - NFmin))

    # 6. Almacenar los datos procesados
    if j == 1:
        datos6 = dato
    else:
        datos9 = dato
 
# 7. Mostrar histogramas de los datos
figure(5)
mpl.hist(datos9, bins='auto', alpha=0.75, rwidth=0.3, color='r', label='num 9')
mpl.hist(datos6, bins='auto', alpha=0.75, rwidth=0.3, color='g', label='num 6')
mpl.legend(loc='upper right')
 
# ======================= Segunda Parte ================================== #
print('Segunda parte del programa: Prueba del sistema')
 
# 8. Esperar y cargar imagen de prueba
time.sleep(2.0)
umbral = 0.5
prueba = array(Image.open(r'D:\Upiita\6to\Patrones\Objetos\prueba69.bmp'))
Ima, Can = label(prueba, structure=con8)
 
# 9. Inicializar matrices A y B
A = zeros_like(Ima)
B = zeros_like(Ima)
 
figure(6)
mpl.imshow(block([[A], [B]]))
 
# 10. Procesar cada objeto en la imagen de prueba
for j in range(1, Can):
    s1 = np.where(Ima == j, 1, 0)
    s2 = binary_fill_holes(s1).astype(int)
    s3 = np.logical_xor(s1, s2)
    Fil1, Col1 = np.nonzero(s1)
    Fil2, Col2 = np.nonzero(s3)
 
    if len(Fil2) == 0:
        continue
 
    objmin, objmax = min(Fil2), max(Fil2)
    NFmin, NFmax = min(Fil1), max(Fil1)
    medio = (objmax - objmin) / 2.0 + objmin
    dato.append((medio - NFmin) / (NFmax - NFmin))
 
    # 11. Clasificar según el umbral
    if dato[-1] > umbral:
        print('Seis')
        A[Fil1, Col1] = 1
        drawnow(repite)
        time.sleep(1)
    else:
        print('Nueve')
        B[Fil1, Col1] = 1
        drawnow(repite)
        time.sleep(1)
 
# ======================= Matriz de Confusión ============================ # Para determinar la eficiencia del sistema
# 12. Entrada de datos de resultados
positivo6 = int(input('Cuántos 6s fueron identificados correctamente: '))
positivo9 = int(input('Cuántos 9s fueron identificados correctamente: '))
 
# 13. Calcular valores negativos
negativo6 = 14.0 - positivo6
negativo9 = 16.0 - positivo9
 
# 14. Crear la matriz de confusión
MatrizConfucion = [[positivo6, negativo6], [positivo9, negativo9]]
 
# 15. Calcular el desempeño del clasificador
Par = ((positivo9 + positivo6) / (16.0 + 14.0)) * 100
print(f'El desempeño del clasificador es = {Par:.2f}%')
print(MatrizConfucion)
# 16. Mostrar la imagen de prueba y los resultados
figure(7)
imshow(prueba)
mpl.show()

#Se calcula la matriz de confusión para determinar el desempeño, si la diagonal principal tiene valores altos es por que la eficiencia puede ser alta, para corroborar se puede utilizar el método F1-score

# Tarea: Para 5 tipografías realizar identificador de Y & T, E & F