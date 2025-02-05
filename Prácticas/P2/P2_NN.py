import pickle
import os
import numpy as np
import pandas as pd


vecte = pd.read_csv(r'D:\Upiita\6to\Patrones\Objetos\Placas base de datos\Numeros\Vector.csv')

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


def guardar_modelo(weights_1, bias_1, weights_2, bias_2, weights_3, bias_3, filename="modelo_nn.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump((weights_1, bias_1, weights_2, bias_2, weights_3, bias_3), f)
    print(f"Modelo guardado en {filename}")

def cargar_modelo(filename="modelo_nn.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            weights_1, bias_1, weights_2, bias_2, weights_3, bias_3 = pickle.load(f)
        print(f"Modelo cargado desde {filename}")
        return weights_1, bias_1, weights_2, bias_2, weights_3, bias_3
    else:
        return None

# Entrenamiento con Backpropagation
def BackPropagation(weights_1, bias_1, weights_2, bias_2, weights_3, bias_3):
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

    for epoch in range(epochs):
        print(f"--- Época {epoch + 1}/{epochs} ---")
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train_one_hot[i:i + batch_size]

            # Forward Propagation
            z1 = np.dot(X_batch, weights_1) + bias_1
            a1 = relu(z1)

            z2 = np.dot(a1, weights_2) + bias_2
            a2 = relu(z2)

            z3 = np.dot(a2, weights_3) + bias_3
            output = softmax(z3)

            loss = categorical_crossentropy(y_batch, output)

            # Backpropagation
            d_z3 = output - y_batch
            d_weights_3 = np.dot(a2.T, d_z3)
            d_bias_3 = d_z3

            d_a2 = np.dot(d_z3, weights_3.T)
            d_z2 = d_a2 * relu_derivative(a2)
            d_weights_2 = np.dot(a1.T, d_z2)
            d_bias_2 = d_z2

            d_a1 = np.dot(d_z2, weights_2.T)
            d_z1 = d_a1 * relu_derivative(a1)
            d_weights_1 = np.dot(X_batch.T, d_z1)
            d_bias_1 = d_z1

            # Actualizar pesos y biases
            weights_3, bias_3 = update_weights(weights_3, bias_3, d_weights_3, d_bias_3, learning_rate)
            weights_2, bias_2 = update_weights(weights_2, bias_2, d_weights_2, d_bias_2, learning_rate)
            weights_1, bias_1 = update_weights(weights_1, bias_1, d_weights_1, d_bias_1, learning_rate)

        print(f"Fin de la época {epoch + 1}/{epochs}, última pérdida: {loss}")

    print("Entrenamiento completado.")
        
# Guardar el modelo después de entrenar
    guardar_modelo(weights_1, bias_1, weights_2, bias_2, weights_3, bias_3)
    return weights_1, bias_1, weights_2, bias_2, weights_3, bias_3

# Declaración de los pesos y bias de la red neuronal de 3 capas
np.random.seed(42)
s1 = 128
s2 = 64
s3 = 10

# Intentar cargar el modelo guardado
modelo_cargado = cargar_modelo()

if modelo_cargado:
    weights_1, bias_1, weights_2, bias_2, weights_3, bias_3 = modelo_cargado
else:
    weights_1 = np.random.randn(40 * 20, s1)
    bias_1 = np.random.randn(s1)

    weights_2 = np.random.randn(s1, s2)
    bias_2 = np.random.randn(s2)

    weights_3 = np.random.randn(s2, s3)
    bias_3 = np.random.randn(s3)

while True:
    weights_1, bias_1, weights_2, bias_2, weights_3, bias_3 = BackPropagation(weights_1, bias_1, weights_2, bias_2, weights_3, bias_3)

    continuar = input("¿Deseas entrenar de nuevo? (s/n): ")
    if continuar.lower() != 's':
        break
