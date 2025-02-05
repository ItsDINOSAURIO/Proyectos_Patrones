from scipy.io import wavfile
from matplotlib import pyplot as plt
from winsound import *

import pyaudio 
import wave
import numpy as np

import pickle
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Función para entrenar la red neuronal con back propagation
def BackPropagation(weights_1,bias_1,weights_2,bias_2,weights_3,bias_3):
    X_train = vecte.iloc[:, 1:].values.astype(float)
    y_train = vecte.iloc[:, 0].values.astype(str)
    encoder = LabelEncoder()
    y_train_int = encoder.fit_transform(y_train)
    # Normalización de los vectores de entrenamiento
    X_train = X_train / np.max(X_train)

    # Parámetros del entrenamiento
    epochs = 10000 #
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
            y_batch = y_train_int[i:i+batch_size]
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


# Cargar los pesos y biases desde el archivo pickle
def cargar_pesos_bias():
    with open('sonido_nn.pkl', 'rb') as f:
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


vecte = pd.read_csv(r"D:\Upiita\6to\Patrones\Practicas\Sonido\sonidos.csv")

plt.close('all')

#Declaracion de los pesos y las bias de la red neuronal de 3 capas
np.random.seed(42)
s1 = 128
s2 = 64
s3 = 3
weights_1 = np.random.randn(150,s1)
bias_1 = np.random.randn(s1)

weights_2 = np.random.randn(s1,s2)
bias_2 = np.random.randn(s2)

weights_3 = np.random.randn(s2,s3)
bias_3 = np.random.randn(s3)

# Verificar si el archivo de pesos ya existe
if os.path.exists('sonido_nn.pkl'):
    print("Cargando pesos y biases guardados...")
    weights_1, bias_1, weights_2, bias_2, weights_3, bias_3 = cargar_pesos_bias()
else:
    print("No se encontraron pesos guardados. Entrenando la red neuronal...")
    weights_1, bias_1, weights_2, bias_2, weights_3, bias_3 = BackPropagation(weights_1, bias_1, weights_2, bias_2, weights_3, bias_3)
    
    # Guardar los pesos y biases para usos futuros
    with open('sonido_nn.pkl', 'wb') as f:
        pickle.dump((weights_1, bias_1, weights_2, bias_2, weights_3, bias_3), f)

while True:
    plt.close('all')
    #Parametros de grabacion 
    formato = pyaudio.paInt16
    canales = 1
    tasa_muestreo = 44100 #Señal mioelectrica 4000  y para audio de alta calidad 44100, mp3 
    #pedazos de 1024 datos de la tasa de muestreo
    tamano_bloque = 1024 # Señal de voz buscar la frecuencia 2 veces arriba de la frecuencia base el muestreo es mejor si se toma mayor a 2 veces la frecuencia base 
    tiempo_grabacion = 3
    nombre_archivo = 'voz.wav'
    #

    #Grabacion
    audio = pyaudio.PyAudio() # Permite generar objetos de audio 
    try: # excepciones del codigo para que no se trabe la maquina, para verificar si esta libre la computadora
        flujo_sonido = audio.open(format = formato, channels = canales, rate = tasa_muestreo, input = True, frames_per_buffer = tamano_bloque)# Abre el objeto, formato, numero de canales, la tasa de muestreo,inicie la entrada, tamaño de buffer
        print('Inicia la grabacion')
        datos_audio = []
        fragmentos = []
        for i in range(0,int(tasa_muestreo/tamano_bloque*tiempo_grabacion)):
            datos_bloque = flujo_sonido.read(tamano_bloque)#Tomando lo que hay en el buffer de la computadora 
            datos_audio.append(datos_bloque)  
            fragmentos.append(np.frombuffer(datos_bloque,dtype = np.int16))
        senal_audio = np.hstack(fragmentos)
        if np.max(np.abs(senal_audio))!=0:
            senal_audio = senal_audio/np.max(np.abs(senal_audio))*32767 #Esto sirve para aumentar la señal ya que esta solamente en un rango de 1 y -1, aumentas el volumen
            senal_audio = senal_audio.astype(np.int16)
        print('Termina la Grabacion')
        flujo_sonido.stop_stream()
        flujo_sonido.close()
    finally:
        audio.terminate()
        
        
    #Elimina los espacios muertos de la señal de audio que se guardo con anterioridad  
    Senal_Normal = senal_audio/np.max(np.abs(senal_audio)) #Normaliza la señal 
    Bin1 = np.where(np.abs(Senal_Normal) >= 0.1,1,0) # Calcula el absoluto lo que no esa mayor a 0.1 sea 0 y si no es asi que sea 1 
    ventana = 440 # Tamaño de la ventana es de 10 miliseg 
    Senal1 = np.convolve(Bin1,np.ones(ventana)/ventana,mode='same')
    Bin2 = np.where(Senal1 >= 0.1,1,0)
    senal_activa = Senal_Normal[Bin2 == 1]


    #...... Filtro de Pre-enfasis enfatiza las altas frecuencias, mejora las frecuencias altas 
    alpha = 0.95 # factor de pre-enfasis 
    Pre = np.roll(senal_activa, 1)- alpha * senal_activa
    Pre[0] = 0 #Primer valor en cero tras el desplazamiento 
    Pre = np.where(np.abs(Pre) >= 0.7, 0, Pre) # Elimina los elementos que son muy agudas 

    #---------------------------------------------------
    frame = 441
    overlap = 397 
    ventanas = range(0,len(Pre) - frame + 1, overlap) # Agrega un translape 
    espectro = []

    for i in ventanas:
        segmento = Pre[i: i+frame] * np.hamming(frame)
        Fourier = np.abs(np.fft.fft(segmento))[:150]
        espectro.append(Fourier)

    #Reconocimiento de patrones 
    #Generar el patron de las palabras 

    comp1 = np.max(np.array(espectro),axis = 0)
    comp2 = np.mean(np.array(espectro),axis = 0)
    patron = (comp1+comp2)/2
    
    res_fin=[]
    z1 = np.dot(patron.T,weights_1) + bias_1
    a1 = relu(z1)

    z2 = np.dot(a1,weights_2) + bias_2
    a2 = relu(z2)

    z3 = np.dot(a2,weights_3) + bias_3
    output = softmax(z3)
    
    # Predicción final: la clase con la probabilidad más alta
    predicted_class = np.argmax(output)
    # Guardar resultados
    if predicted_class == 0:
        predicted_class = "Agua"
        res_fin.append({'prediccion_backprop': predicted_class})
    if predicted_class == 1:
        predicted_class = "Cafe"
        res_fin.append({'prediccion_backprop': predicted_class})  
    if predicted_class == 2:
        predicted_class = "Jugo"
        res_fin.append({'prediccion_backprop': predicted_class})      
    
    
    # Mostrar los resultados finales
    for resultado in res_fin:
        print(f"Predicción (Backpropagation): {resultado['prediccion_backprop']}")
    plt.show()

    continuar = input("¿Deseas decir otra palabra? (s/n): ")
    if continuar.lower() != 's':
        break     