import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import os
from skimage import io, measure, color
from skimage.morphology import disk, closing, dilation, remove_small_objects, area_closing
from skimage.transform import resize
import matplotlib.pyplot as plt
import pickle

#Cargar bases de datos momentos de Hu y Siluetas, carpeta de datos, carpeta específica de los números
hu = pd.read_csv(r'D:\Upiita\6to\Patrones\Objetos\Placas base de datos\Numeros\Momentos_hu.csv')
perf = pd.read_csv(r'D:\Upiita\6to\Patrones\Objetos\Placas base de datos\Numeros\Perfil.csv')
vecte = pd.read_csv(r'D:\Upiita\6to\Patrones\Objetos\Placas base de datos\Numeros\Vector.csv')
carpeta = r'D:\Upiita\6to\Patrones\Objetos\Placas base de datos'

#Definir clases y características a analizar
clases={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
momentosb=hu[[f'Caracteristica {i}'for i in range(1,8)]]
perfilb = perf[[f'Caracteristica {i}' for i in range(1, 81)]]



# Cargar los pesos y biases desde el archivo pickle
def cargar_pesos_bias():
    with open('modelo_nn.pkl', 'rb') as f:
        weights_1, bias_1, weights_2, bias_2, weights_3, bias_3 = pickle.load(f)
    return weights_1, bias_1, weights_2, bias_2, weights_3, bias_3

# Definir funciones de activación y sus respectivas derivadas
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
    
    vectores = []    
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
        # Aplanar el recorte a un vector de 800 elementos
        vector_aplanado = recorte.flatten().astype(int)
        vectores.append(vector_aplanado)
        
        # Dibujar el recuadro de cada región
        rect = plt.Rectangle((columna_min, fila_min), columna_max - columna_min, fila_max - fila_min,
                             edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
    
    plt.axis('off')  
        
    return recortes,vectores

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
# Función para recortar la imagen según la segmentación realizada
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

#Función para transformar la imagen binaria a vector
def vect(imagen):
    vector=imagen.flatten()
    vector=vector/np.max(vector)
    return vector

# Función para calcular la correlación
def correlacion(recorte, clases_p):
    correlaciones = []
    for clase in clases_p:
        corr = np.corrcoef(recorte, clase)[0, 1] 
        correlaciones.append(corr)
    return np.array(correlaciones)

# Función para determinar los centroides de las clases
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

#Función para realizar red neuronal utilizando la librería sklearn
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
#Función para aplicar distintos métodos a las imágenes
def aplic(met):
    recortes_m=[]
    #Aplicar ambos métodos a cada recorte detectado
    for i in range(len(recortes)):
        if met=='hu':
            x=momentos(recortes[i])
        elif met=='perf':
            x=perfil(recortes[i])
        elif met=='vect':
            x=vect(recortes[i])
            
        recortes_m.append(x)
    
    recortes_m=np.array(recortes_m)
    return recortes_m
#Función para entrenar la red neuronal con back propagation
def BackPropagation(weights_1,bias_1,weights_2,bias_2,weights_3,bias_3):
    X_train = vecte.iloc[:, 1:].values.astype(int)

    y_train = vecte.iloc[:, 0].values.astype(int)
    y_train_one_hot = np.eye(10)[y_train]  # Convertir la etiqueta a one-hot encoding

    # Normalización de los vectores de entrenamiento
    X_train = X_train / 255.0

    # Parámetros del entrenamiento
    epochs = 1000 #
    # learning_rate = 0.5
    learning_rate=0.001
    batch_size = 1

    # Entrenamiento con Backpropagation
    for epoch in range(epochs):
        # learning_rate-=0.0001
        print(f"--- Época {epoch+1}/{epochs} ---") 
        # print(f'La tasa de aprendizaje es: {learning_rate}')
        for i in range(0, X_train.shape[0], batch_size):
            # Extraer un mini-batch
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train_one_hot[i:i+batch_size]
            # print(f"Lote {batch_count}: procesando una imagen")

            # Forward Propagation
            # Capa 1
            z1 = np.dot(X_batch,weights_1) + bias_1
            a1 = relu(z1)

            # Capa 2
            z2 = np.dot(a1,weights_2) + bias_2
            a2 = relu(z2)

            # Capa de salida
            z3 = np.dot(a2,weights_3) + bias_3
            output = softmax(z3)

            # Calcular la pérdida (categorical crossentropy)
            loss = categorical_crossentropy(y_batch, output)
            # print(f"Pérdida: {loss}")

            # Backpropagation
            # Gradiente de la capa de salida
            d_z3 = output - y_batch  # Derivada del error con respecto a z3
            d_weights_3 = np.dot(a2.T,d_z3)
            d_bias_3 = d_z3

            # Gradiente de la segunda capa oculta
            d_a2 = np.dot(d_z3, weights_3.T)
            d_z2 = d_a2 * relu_derivative(a2)
            d_weights_2 = np.dot(a1.T,d_z2)
            d_bias_2 = d_z2

            # Gradiente de la primera capa oculta
            d_a1 = np.dot(d_z2, weights_2.T)
            d_z1 = d_a1 * relu_derivative(a1)
            d_weights_1 = np.dot(X_batch.T,d_z1)
            d_bias_1 = d_z1

            # Actualizar pesos y biases
            weights_3, bias_3 = update_weights(weights_3, bias_3, d_weights_3, d_bias_3, learning_rate)
            weights_2, bias_2 = update_weights(weights_2, bias_2, d_weights_2, d_bias_2, learning_rate)
            weights_1, bias_1 = update_weights(weights_1, bias_1, d_weights_1, d_bias_1, learning_rate)



        # Mostrar el progreso después de cada época
        print(f"Fin de la época {epoch+1}/{epochs}, última pérdida: {loss}")

    print("Entrenamiento completado.")
    return weights_1,bias_1,weights_2,bias_2,weights_3,bias_3


#Centroides de las clases 0-9
clases_h=cla('hu')
clases_p=cla('perf')

#Definir y entrenar Red neuronal
NN,escalador=red_hu()

#Declaracion de los pesos y las bias de la red neuronal de 3 capas
np.random.seed(42)
s1 = 128
s2 = 64
s3=10
weights_1 = np.random.randn(40*20,s1)
bias_1 = np.random.randn(s1)

weights_2 = np.random.randn(s1,s2)
bias_2 = np.random.randn(s2)

weights_3 = np.random.randn(s2,s3)
bias_3 = np.random.randn(s3)

# Verificar si el archivo de pesos ya existe
if os.path.exists('modelo_nn.pkl'):
    print("Cargando pesos y biases guardados...")
    weights_1, bias_1, weights_2, bias_2, weights_3, bias_3 = cargar_pesos_bias()
else:
    print("No se encontraron pesos guardados. Entrenando la red neuronal...")
    weights_1, bias_1, weights_2, bias_2, weights_3, bias_3 = BackPropagation(weights_1, bias_1, weights_2, bias_2, weights_3, bias_3)
    
    # Guardar los pesos y biases para usos futuros
    with open('modelo_nn.pkl', 'wb') as f:
        pickle.dump((weights_1, bias_1, weights_2, bias_2, weights_3, bias_3), f)

# while True:
#     weights_1,bias_1,weights_2,bias_2,weights_3,bias_3=BackPropagation(weights_1,bias_1,weights_2,bias_2,weights_3,bias_3)
    

while True:
    plt.close('all')
    # Elegir la imagen y recortarla
    imagen = elegir(carpeta)
    recortes,digitos=recortar(imagen)

    #Aplicar métodos a las regiones encontradas
    recorte_hu=aplic('hu')
    recorte_perf=aplic('perf')
    recorte_vect=aplic('vect')

    res_fin=[]

    for i in range(len(digitos)):  
        # Realizar predicción usando el modelo entrenado (forward propagation)
        hu_recorte=recorte_hu[i].reshape(1,-1)
        co_NN=NN.predict(escalador.transform(hu_recorte))
        
        # Clasificación por correlación para el Perfil
        perfil_recorte=recorte_perf[i]
        correlaciones_perf = correlacion(perfil_recorte, clases_p)
        imax_corr_perf = np.argmax(correlaciones_perf) 
        max_corr_perf = correlaciones_perf[imax_corr_perf]

        # Asignar la clase con mayor correlación
        if max_corr_perf > 0.99:  
            co_perf = clases[str(imax_corr_perf)]
        else:
            co_perf = 'Desconocida'
        
        vector_digito =  recorte_vect[i].astype(int)  
        # print(vector_digito)
        
        z1 = np.dot(vector_digito.T,weights_1) + bias_1
        a1 = relu(z1)

        z2 = np.dot(a1,weights_2) + bias_2
        a2 = relu(z2)

        z3 = np.dot(a2,weights_3) + bias_3
        output = softmax(z3)
        # print(output)
        
        # Predicción final: la clase con la probabilidad más alta
        predicted_class = np.argmax(output)

        # Filtro para la correlación
        if max_corr_perf >= 0.99:
            resultado_correlacion = clases[str(np.argmax(correlaciones_perf))]
        else:
            resultado_correlacion = 'Desconocida'

        # Guardar resultados
        res_fin.append({
            'recorte': i + 1,
            'prediccion_backprop': predicted_class,
            'clasificacion_hu': co_NN[0],
            'clasificacion_correlacion': resultado_correlacion
        })

    # Mostrar los resultados finales
    for resultado in res_fin:
        print(f"Recorte {resultado['recorte']} - "
            f"Predicción (Backpropagation): {resultado['prediccion_backprop']}, "
            f"Clasificación (Hu): {resultado['clasificacion_hu']}, "
            f"Clasificación (Correlación): {resultado['clasificacion_correlacion']}")
    plt.show()

    continuar = input("¿Deseas elegir otra imagen? (s/n): ")
    if continuar.lower() != 's':
        break 
    # continuar = input("¿Deseas entrenar de nuevo? (s/n): ")
    # if continuar.lower() != 's':
    #     break 