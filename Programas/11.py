from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
import os
 
# Directorio de la carpeta que contiene las imágenes de los dígitos
input_dir = r"C:\Users\PSC54195\Documents\UPIITA_2023_1\Protocolos\digits_database"
 
# Tamaño al que queremos redimensionar las imágenes (28x28 píxeles)
image_size = (28, 28)
 
# Listas para almacenar los patrones de entrada y las etiquetas
patterns = []
labels = []
 
# Función para leer y procesar las imágenes
def process_images(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # Leer la imagen
            image_path = os.path.join(input_folder, filename)
            image = io.imread(image_path, as_gray=True)
            # Redimensionar la imagen a 28x28 píxeles
            resized_image = transform.resize(image, image_size, mode='reflect', anti_aliasing=True)
            # Convertir la imagen en un vector unidimensional
            image_vector = resized_image.flatten()
            # Normalizar los valores de los píxeles (entre 0 y 1)
            image_vector = image_vector / np.max(image_vector)
            # Añadir el vector a la lista de patrones
            patterns.append(image_vector)
            # Extraer la etiqueta del nombre del archivo (asumiendo que contiene información sobre el número)
            # Por ejemplo, si el archivo es 'row_1_digit_0.png', extraemos el número '0'
            label = int(filename.split('_')[-1].split('.')[0])
            labels.append(label)
 
# Procesar las imágenes en la carpeta
process_images(input_dir)
 
# Convertir las listas a arrays de numpy para ser usados en la red neuronal
patterns   = np.array(patterns)
labels     = np.array(labels)
labels[0]  = 5
labels[1]  = 4
labels[2]  = 8
labels[3]  = 9
labels[4]  = 3
labels[5]  = 7
labels[6]  = 6
labels[7]  = 0
labels[8]  = 2
labels[9]  = 1
labels[10] = 7
labels[11] = 4
labels[12] = 8
labels[13] = 3
labels[14] = 5
labels[15] = 2
labels[16] = 6
labels[17] = 1
labels[18] = 9
labels[19] = 0
labels[20] = 9
labels[21] = 7
labels[22] = 8
labels[23] = 5
labels[24] = 4
labels[25] = 3
labels[26] = 6
labels[27] = 2
labels[28] = 1
labels[29] = 0
labels[30] = 8
labels[31] = 9
labels[32] = 6
labels[33] = 4
labels[34] = 5
labels[35] = 7
labels[36] = 1
labels[37] = 3
labels[38] = 2
labels[39] = 0
labels[40] = 9
labels[41] = 8
labels[42] = 7
labels[43] = 6
labels[44] = 4
labels[45] = 5
labels[46] = 2
labels[47] = 3
labels[48] = 1
labels[49] = 0
 
# Mostrar información sobre los datos procesados
print(f"Total de patrones extraídos: {len(patterns)}")
print(f"Tamaño de cada patrón: {patterns.shape[1]} (corresponde a la imagen redimensionada a 28x28 = 784 píxeles)")
print(f"Etiquetas de los primeros 10 patrones: {labels[:10]}")
 
# Extraer la primera fila de la variable 'patterns'
first_pattern = patterns[48]
 
# Darle forma a la imagen (reshape) de 28x28 píxeles
reconstructed_image = first_pattern.reshape((28, 28))
 
# Mostrar la imagen reconstruida
plt.close('all')
plt.imshow(reconstructed_image, cmap='gray')
plt.title(f'Número representado: {labels[48]}')  # Mostrar la etiqueta correspondiente
plt.axis('off')
plt.show()
 
#=================================================================#
#<================ Funciones de Activación ======================>#
def hardlim(n):
    if (n > 0):
        a = 1
    else:
        a = 0
    return a
 
def purelin(n):
    a = 1 * n
    return a
 
# Derivada de la Funcion purelin
def purelin_derivada(n):
    f_punto = 1
    return f_punto
 
def sigmoid(n):
    f = 1 / (1 + np.exp(-n))
    return f
 
# Derivada de la Funcion sigmoid
def sigmoid_derivada(n):
    f_punto = n * (1 - n)
    return f_punto
 
# Función de activación
def softmax(n):
    exps   = np.exp(n - np.max(n, axis=1, keepdims=True))
    output = exps / np.sum(exps, axis=1, keepdims=True)
    return output
 
# Inicialización de los pesos y sesgos
def initialize_weights(input_size, hidden1_size, hidden2_size, output_size):
    np.random.seed(42)
    weights = {
        'W1': np.random.randn(input_size,   hidden1_size) * 0.01,
        'b1': np.zeros((1, hidden1_size)),
        'W2': np.random.randn(hidden1_size, hidden2_size) * 0.01,
        'b2': np.zeros((1, hidden2_size)),
        'W3': np.random.randn(hidden2_size, output_size)  * 0.01,
        'b3': np.zeros((1, output_size)),
    }
    return weights
 
# Función de forward propagation
def forward_propagation(X, weights):
    n1 = np.dot(X, weights['W1']) + weights['b1']
    a1 = sigmoid(n1)
    n2 = np.dot(a1, weights['W2']) + weights['b2']
    a2 = sigmoid(n2)
    n3 = np.dot(a2, weights['W3']) + weights['b3']
    a3 = softmax(n3)
    cache = {'a1': a1, 'a2': a2, 'a3': a3, 'n1': n1, 'n2': n2, 'n3': n3}
    return a3, cache
 
# Función de cálculo de pérdida (cross-entropy)
def compute_loss(Y, a3):
    m = Y.shape[0]
    log_probs = -np.log(a3[range(m), Y.argmax(axis=1)])
    loss = np.sum(log_probs) / m
    return loss
 
# Función de backpropagation
def back_propagation(X, Y, cache, weights):
    m = X.shape[0]
    # Cálculo de los gradientes
    dn3 = cache['a3'] - Y
    dW3 = np.dot(cache['a2'].T, dn3) / m
    db3 = np.sum(dn3, axis=0, keepdims=True) / m
 
    dn2 = np.dot(dn3, weights['W3'].T) * sigmoid_derivada(cache['a2'])
    dW2 = np.dot(cache['a1'].T, dn2) / m
    db2 = np.sum(dn2, axis=0, keepdims=True) / m
 
    dn1 = np.dot(dn2, weights['W2'].T) * sigmoid_derivada(cache['a1'])
    dW1 = np.dot(X.T, dn1) / m
    db1 = np.sum(dn1, axis=0, keepdims=True) / m
 
    # Actualización de los pesos y sesgos
    gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}
    return gradients
 
# Actualización de los pesos usando los gradientes calculados
def update_weights(weights, gradients, learning_rate):
    weights['W1'] -= learning_rate * gradients['dW1']
    weights['b1'] -= learning_rate * gradients['db1']
    weights['W2'] -= learning_rate * gradients['dW2']
    weights['b2'] -= learning_rate * gradients['db2']
    weights['W3'] -= learning_rate * gradients['dW3']
    weights['b3'] -= learning_rate * gradients['db3']
    return weights
 
# Función de entrenamiento
def train(X, Y, input_size, hidden1_size, hidden2_size, output_size, learning_rate, epochs):
    weights = initialize_weights(input_size, hidden1_size, hidden2_size, output_size)
    for epoch in range(epochs):
        # Forward propagation
        a3, cache = forward_propagation(X, weights)
 
        # Compute loss
        loss = compute_loss(Y, a3)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
 
        # Backpropagation
        gradients = back_propagation(X, Y, cache, weights)
 
        # Update weights
        weights = update_weights(weights, gradients, learning_rate)
 
    return weights
 
# Predicción
def predict(X, weights):
    a3, _ = forward_propagation(X, weights)
    return np.argmax(a3, axis=1)
 
# Parámetros de la red
input_size    = 784
hidden1_size  = 128
hidden2_size  = 64
output_size   = 10  # Cambiado a 10 para los dígitos 0-9
learning_rate = 0.01
epochs        = 35000
 
# Convertir las etiquetas a one-hot encoding
Y_one_hot = np.zeros((labels.size, output_size))
Y_one_hot[np.arange(labels.size), labels] = 1
 
# Entrenar la red neuronal
weights = train(patterns, Y_one_hot, input_size, hidden1_size, hidden2_size, output_size, learning_rate, epochs)
 
# Evaluar el modelo en los datos de entrenamiento
predictions = predict(patterns, weights)
accuracy    = np.mean(predictions == labels)
print(f'Precisión en el conjunto de entrenamiento: {accuracy:.4f}')
 
#=================================================================#
#<===============================================================>#
# Directorio de la carpeta que contiene las imágenes de los dígitos
input_dir_test = r"test_database"
 
# Tamaño al que queremos redimensionar las imágenes (28x28 píxeles)
image_size_test = (28, 28)
 
# Listas para almacenar los patrones de entrada y las etiquetas
patterns_test = []
labels_test   = []
 
# Función para leer y procesar las imágenes (reutilizable para el conjunto de prueba)
def process_images(input_folder):
    patterns_list = []
    labels_list = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # Leer la imagen
            image_path = os.path.join(input_folder, filename)
            image = io.imread(image_path, as_gray=True)
            # Redimensionar la imagen a 28x28 píxeles
            resized_image = transform.resize(image, image_size_test, mode='reflect', anti_aliasing=True)
            # Convertir la imagen en un vector unidimensional
            image_vector = resized_image.flatten()
            # Normalizar los valores de los píxeles (entre 0 y 1)
            image_vector = image_vector / np.max(image_vector)
            # Añadir el vector a la lista de patrones
            patterns_list.append(image_vector)
            # Extraer la etiqueta del nombre del archivo (asumiendo que contiene información sobre el número)
            label = int(filename.split('_')[-1].split('.')[0])
            labels_list.append(label)
 
    return np.array(patterns_list), np.array(labels_list)
 
# Procesar las imágenes de prueba
patterns_test, labels_test = process_images(input_dir_test)
 
# Establecer las etiquetas manualmente si es necesario (usando las etiquetas que se proporcionaron)
labels_test[0]  = 8
labels_test[1]  = 6
labels_test[2]  = 4
labels_test[3]  = 0
labels_test[4]  = 9
labels_test[5]  = 1
labels_test[6]  = 7
labels_test[7]  = 5
labels_test[8]  = 2
labels_test[9]  = 3
labels_test[10] = 8
labels_test[11] = 9
labels_test[12] = 6
labels_test[13] = 7
labels_test[14] = 5
labels_test[15] = 4
labels_test[16] = 3
labels_test[17] = 1
labels_test[18] = 2
labels_test[19] = 0
labels_test[20] = 8
labels_test[21] = 9
labels_test[22] = 7
labels_test[23] = 5
labels_test[24] = 6
labels_test[25] = 4
labels_test[26] = 3
labels_test[27] = 2
labels_test[28] = 0
labels_test[29] = 1
 
# Predicción en el conjunto de prueba
predictions_test = predict(patterns_test, weights)
 
# Calcular la precisión en el conjunto de prueba
accuracy_test = np.mean(predictions_test == labels_test)
print(f'Precisión en el conjunto de prueba: {accuracy_test:.4f}')
 
# Mostrar algunas imágenes del conjunto de prueba junto con las etiquetas predichas
num_images_to_show = 20
plt.figure(figsize=(15, 6))  # Ajustar el tamaño de la figura para acomodar dos filas
 
for i in range(num_images_to_show):
    # Para la primera fila (imágenes 0 a 9)
    if i < 10:
        plt.subplot(2, 10, i + 1)  # 2 filas, 10 columnas
    # Para la segunda fila (imágenes 10 a 19)
    else:
        plt.subplot(2, 10, i + 1)
    # Mostrar la imagen
    image = patterns_test[i].reshape((28, 28))
    plt.imshow(image, cmap='gray')
    plt.title(f'Verdadero: {labels_test[i]}\nPredicho: {predictions_test[i]}')
    plt.axis('off')
 
plt.tight_layout()
plt.show()