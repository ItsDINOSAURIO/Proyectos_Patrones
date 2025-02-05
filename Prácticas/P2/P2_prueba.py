import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import os
from skimage import io, measure, color
from skimage.morphology import disk, closing, dilation, remove_small_objects, area_closing
from skimage.transform import resize
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

#Cargar bases de datos momentos de Hu y Siluetas, carpeta de datos, carpeta específica de los números
hu = pd.read_csv(r'D:\Upiita\6to\Patrones\Objetos\Placas base de datos\Numeros\Momentos_hu.csv')
perf = pd.read_csv(r'D:\Upiita\6to\Patrones\Objetos\Placas base de datos\Numeros\Perfil.csv')
carpeta = r'D:\Upiita\6to\Patrones\Objetos\Placas base de datos'
train_dir = r'D:\Upiita\6to\Patrones\Objetos\Placas base de datos\Numeros'

#Definir clases y características a analizar
clases={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
momentosb=hu[[f'Caracteristica {i}'for i in range(1,8)]]
perfilb = perf[[f'Caracteristica {i}' for i in range(1, 81)]]

# Definir funciones de activación
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x)) 
    if x.ndim == 1:
        return exp_x / exp_x.sum()  # Para vectores unidimensionales
    else:
        return exp_x / exp_x.sum(axis=1, keepdims=True)  # Para matrices 2D

# Función de pérdida: Categorical Crossentropy
def categorical_crossentropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9))  # Añadir epsilon para estabilidad numérica

# Backpropagation
def update_weights(weights, biases, d_weights, d_biases, learning_rate):
    weights -= learning_rate * d_weights
    biases -= learning_rate * np.sum(d_biases, axis=0)  # Asegurar que el bias tenga la forma correcta
    return weights, biases

#Función para Centroide de clase
def carac(val,met):
    if met == 'hu':
        car = val[[f'Caracteristica {i}'for i in range(1,8)]].values
        prom = np.mean(car, axis=0)
    elif met=='perf':
        car = val[[f'Caracteristica {i}' for i in range(1, 81)]].values
        prom = np.mean(car, axis=0)
    
    return prom

# Función de filtrado
def gbf(imagen):
    gris = color.rgb2gray(imagen)
    estructura = disk(4.8)
    binario = (gris < 0.3)
    binario = closing(binario, estructura)
    binario = dilation(binario, estructura)
    binario = remove_small_objects(binario, min_size=100)
    binario = area_closing(binario, area_threshold=450)
    return binario

# Identificar regiones, recortar y mostrar
def ir(imagen):
    binario = gbf(imagen)
    etiquetas, _ = measure.label(binario, return_num=True, connectivity=2)
    propiedades = measure.regionprops(etiquetas)
    
    vectores_aplanados = []    
    recortes = []
    
    # Crear la figura para mostrar la imagen filtrada con los recuadros de las regiones
    _, ax = plt.subplots()
    ax.imshow(binario, cmap='gray')
    
    # Recorrer las regiones detectadas
    for region in propiedades:
        fila_min, columna_min, fila_max, columna_max = region.bbox
        recorte = binario[fila_min:fila_max, columna_min:columna_max]
        recorte = resize(recorte, (40, 20), anti_aliasing=False)  
        recortes.append(recorte)
        recorte_neu = resize(recorte, (28, 28), anti_aliasing=False) 
        
        # Aplanar el recorte a un vector de 784 elementos
        vector_aplanado = recorte_neu.flatten().astype(int)
        vectores_aplanados.append(vector_aplanado)
        
        # Dibujar el recuadro de cada región
        rect = plt.Rectangle((columna_min, fila_min), columna_max - columna_min, fila_max - fila_min,
                             edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
    
    plt.axis('off')  
        
    return recortes,vectores_aplanados

# Función para elegir una imagen arbitrariamente
def elegir(carpeta):
    imagenes = [f for f in os.listdir(carpeta) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    # Mostrar las imágenes disponibles
    print("Imágenes disponibles:")
    for idx, nombre in enumerate(imagenes):
        print(f"{idx}: {nombre}")
    
    # Solicitar la selección de la imagen
    seleccion = int(input(f"Selecciona el número de la imagen (0-{len(imagenes)-1}): "))
    if seleccion < 0 or seleccion >= len(imagenes):
        print("Selección inválida.")
        return None
    
    # Cargar la imagen seleccionada
    ruta = os.path.join(carpeta, imagenes[seleccion])
    imagen = io.imread(ruta)
    
    return imagen

def recortar(imagen):
    #Determinar el número de regiones de la imagen de entrada
    if imagen is not None:
        recortes, digitos = ir(imagen)
        print(f"Se encontraron {len(recortes)} secciones.")
    else:
        print("No se seleccionó ninguna imagen.")
    
    return recortes,digitos

#Función para el cálculo de momentos de hu
def momentos(imagen):
    imagen=imagen.astype(np.uint8)*255
    mu = measure.moments(imagen)
    nu = measure.moments_normalized(mu)
    hu = measure.moments_hu(nu)  
    return hu

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

# Función para calcular la correlación
def correlacion(recorte, clases_p):
    correlaciones = []
    for clase in clases_p:
        corr = np.corrcoef(recorte, clase)[0, 1] 
        correlaciones.append(corr)
    return np.array(correlaciones)

def cla(met):
    clases_f=[]
    for i in range(10):
        if met=='hu':  
            val = hu[hu['Clases'] == i]
            centroides = carac(val,'hu')
            clases_f.append(centroides)
        elif met=='perf':
            val = perf[perf['Clases'] == i]
            centroides = carac(val,'perf')
            clases_f.append(centroides)
    return np.array(clases_f)

def red_hu():

    #Definición de parametros para la red neuronal
    X=hu.drop(columns='Clases').values
    Y=hu['Clases'].values
    escalador = StandardScaler()
    X = escalador.fit_transform(X)

    # Entrenamiento y definición de la red neuronal
    NN = MLPClassifier(hidden_layer_sizes=(40), 
                        activation='relu', 
                        solver='adam', 
                        max_iter=100000, 
                        random_state=42,
                        learning_rate='constant',
                        learning_rate_init=0.01)
    NN.fit(X,Y)
    return NN, escalador

def aplic(met):
    recortes_m=[]
    #Aplicar ambos métodos a cada recorte detectado
    for i in range(len(recortes)):
        if met=='hu':
            x=momentos(recortes[i])
        elif met=='perf':
            x=perfil(recortes[i])
            
        recortes_m.append(x)
    
    recortes_m=np.array(recortes_m)
    return recortes_m

# Elegir la imagen y recortarla
imagen = elegir(carpeta)
recortes,digitos=recortar(imagen)

#Centroides de las clases 0-9
clases_h=cla('hu')
clases_p=cla('perf')

#Aplicar ambos métodos a las regiones encontradas
recorte_hu=aplic('hu')
recorte_perf=aplic('perf')

#Definir y entrenar Red neuronal
NN,escalador=red_hu()

# Definir el tamaño de las imágenes que queremos (28x28)
image_size = (28, 28)

# Verificar si el directorio existe
if not os.path.exists(train_dir):
    print(f"El directorio {train_dir} no existe. Asegúrate de que la ruta es correcta.")
else:
    # Crear generadores de imágenes para cargar las imágenes desde las carpetas
    train_datagen = ImageDataGenerator(rescale=1./255)  # Normalización

    # Cargar las imágenes de las carpetas de entrenamiento
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        color_mode="grayscale",   # En caso de que las imágenes sean en escala de grises
        batch_size=1,             # Procesar una imagen a la vez para simplificar el forward propagation
        class_mode='sparse'
    )

    # Verificar si se cargaron imágenes
    if train_generator.samples == 0:
        print("No se encontraron imágenes en el directorio.")
    else:

        #Declaracion de los pesos y las bias de la red neuronal de 3 capas
        np.random.seed(42)
        weights_1 = np.random.randn(28*28, 170)
        bias_1 = np.random.randn(170)

        weights_2 = np.random.randn(170, 60)
        bias_2 = np.random.randn(60)

        weights_3 = np.random.randn(60, 10)
        bias_3 = np.random.randn(10)

        # Parámetros del entrenamiento
        epochs = 1000
        learning_rate = 0.001

        # Entrenamiento con Backpropagation
        for epoch in range(epochs):
            print(f"--- Época {epoch+1}/{epochs} ---") 
            batch_count = 0  # Contador de lotes

            for x_train, y_train in train_generator:
                batch_count += 1
                print(f"Lote {batch_count}: procesando una imagen")

                # Forward Propagation
                x_train = x_train.reshape((1, 28*28))  # Aplanar la imagen
                y_train_one_hot = np.eye(10)[int(y_train[0])]  # Convertir la etiqueta a one-hot encoding

                # Capa 1
                z1 = np.dot(x_train, weights_1) + bias_1
                a1 = relu(z1)

                # Capa 2
                z2 = np.dot(a1, weights_2) + bias_2
                a2 = relu(z2)

                # Capa de salida
                z3 = np.dot(a2, weights_3) + bias_3
                output = softmax(z3)

                # Calcular la pérdida (categorical crossentropy)
                loss = categorical_crossentropy(y_train_one_hot, output)
                print(f"Pérdida: {loss}")

                # Backpropagation
                # Gradiente de la capa de salida
                d_z3 = output - y_train_one_hot  # Derivada del error con respecto a z3
                d_weights_3 = np.dot(a2.T, d_z3)
                d_bias_3 = d_z3

                # Gradiente de la segunda capa oculta
                d_a2 = np.dot(d_z3, weights_3.T)
                d_z2 = d_a2 * relu_derivative(z2)
                d_weights_2 = np.dot(a1.T, d_z2)
                d_bias_2 = d_z2

                # Gradiente de la primera capa oculta
                d_a1 = np.dot(d_z2, weights_2.T)
                d_z1 = d_a1 * relu_derivative(z1)
                d_weights_1 = np.dot(x_train.T, d_z1)
                d_bias_1 = d_z1

                # Actualizar pesos y biases
                weights_3, bias_3 = update_weights(weights_3, bias_3, d_weights_3, d_bias_3, learning_rate)
                weights_2, bias_2 = update_weights(weights_2, bias_2, d_weights_2, d_bias_2, learning_rate)
                weights_1, bias_1 = update_weights(weights_1, bias_1, d_weights_1, d_bias_1, learning_rate)

                # Salir del ciclo después de procesar un lote para depuración
                if batch_count == 1:
                    break

            # Mostrar el progreso después de cada época
            print(f"Fin de la época {epoch+1}/{epochs}, última pérdida: {loss}")

        print("Entrenamiento completado.")

for i in range(len(digitos)):  
    vector_digito = digitos[i]  
    
    z1 = np.dot(vector_digito, weights_1) + bias_1
    a1 = relu(z1)

    z2 = np.dot(a1, weights_2) + bias_2
    a2 = relu(z2)

    z3 = np.dot(a2, weights_3) + bias_3
    output = softmax(z3)
    
    # Predicción final: la clase con la probabilidad más alta
    predicted_class = np.argmax(output)
    print(f"Recorte {i + 1} - El modelo predice que el número es: {predicted_class}")

# Clasificación por momentos de hu y perfil
for hu_recorte, perfil_recorte in zip(recorte_hu, recorte_perf):
    # Realizar predicción usando el modelo entrenado (forward propagation)

    hu_recorte=hu_recorte.reshape(1,-1)
    co_NN=NN.predict(escalador.transform(hu_recorte))
    print(f'Clasificación (Hu): {co_NN[0]}')

    # Clasificación por correlación para el Perfil
    correlaciones_perf = correlacion(perfil_recorte, clases_p)
    imax_corr_perf = np.argmax(correlaciones_perf) 
    max_corr_perf = correlaciones_perf[imax_corr_perf]

    # Asignar la clase con mayor correlación
    if max_corr_perf > 0.99:  
        co_perf = clases[str(imax_corr_perf)]
    else:
        co_perf = 'Desconocida'
    
    print(f'Clasificación (Perfil): {co_perf}')
plt.show()