import wave
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import os
import serial
import time
from scipy import signal
import pickle
from winsound import PlaySound, SND_FILENAME, SND_ASYNC
import pyaudio
import wave
import scipy.io.wavfile as wav

# plt.close('all')

#Parámetros iniciales
formato = pyaudio.paInt16
canales = 1
tasa_muestreo = 44100
tamano_bloque = 1024
tiempo_grabacion = 2

# Referencias
# en = ["forward", "backward", "left", "right"]
en = ["norte", "sur", "este", "oeste"]
es = ["adelante", "atras", "izquierda", "derecha"]
# com = ("izquierda", "adelante", "11", "derecha", "00", "atras","10", "01")#('00','11', '10', '01')
com = ('01','00', '10', '11')


#FUnción para agregar nuevos audios a la base de datos
def add_db(tipo, muestra):
    directorio_base = r"D:\Upiita\6to\Patrones\Proyectos\Carrito\audios\train"
    if not os.path.exists(directorio_base):
        os.makedirs(directorio_base)

    # Crear la subcarpeta para el tipo de audio si no existe
    subcarpeta = os.path.join(directorio_base, tipo)
    if not os.path.exists(subcarpeta):
        os.makedirs(subcarpeta)

    nombre_archivo = f"{tipo}_{muestra:02}.wav"
    ruta_archivo = os.path.join(subcarpeta, nombre_archivo)

    grabar_audio(ruta_archivo)

#Función para grabar audios dependiendo de los dispositivos disponibles
def grabar_audio(nombre_archivo):

    audio = pyaudio.PyAudio()
    print("Dispositivos disponibles:")
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        print(f"ID: {i}, Nombre: {info['name']}")
    index_selected = int(input("Introduce el ID del micrófono a utilizar: "))

    try:
        flujo_sonido = audio.open(format=formato, channels=canales, rate=tasa_muestreo,
                                  input=True, frames_per_buffer=tamano_bloque, input_device_index=index_selected)
        print('Inicia la grabación')
        datos_audio = []
        fragmentos = []
        for i in range(0, int(tasa_muestreo / tamano_bloque * tiempo_grabacion)):
            datos_bloque = flujo_sonido.read(tamano_bloque, exception_on_overflow=False)
            datos_audio.append(datos_bloque)
            fragmentos.append(np.frombuffer(datos_bloque, dtype=np.int16))
        senal_audio = np.hstack(fragmentos)
        print('Termina la Grabación')

        if np.max(np.abs(senal_audio)) != 0:
            senal_audio = senal_audio / np.max(np.abs(senal_audio)) * 32767
            senal_audio = senal_audio.astype(np.int16)

        flujo_sonido.stop_stream()
        flujo_sonido.close()
    except Exception as e:
        print(f"Error durante la grabación: {e}")
        return None
    finally:
        audio.terminate()

    try:
        with wave.open(nombre_archivo, 'wb') as archivo_wave:
            archivo_wave.setnchannels(canales)
            archivo_wave.setsampwidth(audio.get_sample_size(formato))
            archivo_wave.setframerate(tasa_muestreo)
            archivo_wave.writeframes(b''.join([senal_audio.tobytes()]))
    except Exception as e:
        print(f"Error al guardar el archivo WAV: {e}")
        return None
    
    PlaySound(nombre_archivo,SND_FILENAME|SND_ASYNC)


    return senal_audio

#Caso contrario; Función para cargar un audio pregrabado
def cargar_audio(file_path):
    FS = 44100
    with wave.open(file_path, 'rb') as wav_file:
        params = wav_file.getparams()
        _, sample_width, frame_rate, n_frames, _, _ = params[:6]
        frames = wav_file.readframes(n_frames)
    dtype = np.int16 if sample_width == 2 else np.int32
    data = np.frombuffer(frames, dtype=dtype)
    if frame_rate != FS:
        rr = FS / frame_rate
        frame_rate = FS
        data = signal.resample(data, int(len(data) * rr))
    return data, frame_rate

#Función para enviar la instrucción al arduino
def sarduino(resultado):
    arduino.write(resultado.encode() + b'\n')
    return

#Función para entrenar la red
def Compet_Recur_Tr(patrones, W):
    pkl = 'pesos_bias.pkl'
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
            W = data['W']
            bias = data['bias']
        print("Pesos y bias cargados desde el archivo pkl.")
    else:
        print("No se encontró el archivo pkl, se utilizarán los pesos iniciales.")
        bias = np.zeros(W.shape[0])

    epocas = 10
    alpha = 0.0001

    for epoca in range(epocas):
        for i in range(patrones.shape[0]):
            dists=np.linalg.norm(W - patrones[i, :], axis=1)
            # a1 = np.dot(W, patrones[i, :].T) + bias #
            ganador = np.argmin(dists)
            for j in range(W.shape[0]):
                if j == ganador:
                    W[j, :] += alpha * (patrones[i, :] - W[j, :])
            bias[ganador] -= 0.2 * (1 + bias[ganador])
        # alpha*=0.9
        # print(W)
        if (epoca)%100==0:
            print(f"Época: {epoca}")
        with open(pkl, 'wb') as f:
            pickle.dump({'W': W, 'bias': bias}, f)

#Función para probar la red
def Compet_Recur_Ts(patron):
    pkl = 'pesos_bias.pkl'#r"C:\Users\emili\Downloads\pesos_bias.pkl"#
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
            W = data['W']
            bias = data['bias']
        print("Pesos y bias cargados desde el archivo pkl.")
    else:
        print("No se encontró el archivo pkl, favor de entrenar la red.")
        return

    epsilon = 0.01
    iteracion = 100
    print(patron)

    # Iteración para el proceso de salida
    a1 = np.dot(W, patron) + bias
    a2 = a1.copy()
    for _ in range(iteracion):
        new_a2 = np.zeros_like(a2)
        for i in range(len(a2)):
            ini = epsilon * (np.sum(a2) - a2[i])
            new_a2[i] = max(0, a2[i] - ini)
        a2 = new_a2

    winner_ind = np.argmax(a2)
    print(f"Salida final: {a2}")
    print(f"Combinación: {com[winner_ind]}")

    # Graficar las señales
    plt.figure(figsize=(10, 6))

    # Crear un subplot para cada señal
    for i in range(W.shape[0]):
        plt.subplot(5, 1, i+1)  # 4 filas y 1 columna
        plt.plot(W[i, :])
        plt.title(f'Señal {i+1}')
        plt.xlabel('Muestras')
        plt.ylabel('Amplitud')

    plt.subplot(5,1,5)
    plt.title(f'Señal patron')
    plt.plot(patron)

    plt.tight_layout()
    # plt.show()

    return com[winner_ind]

def DTW(test_feats, win_dur=0.02):
    pkl = 'pesos_bias.pkl'#r"C:\Users\emili\Downloads\pesos_bias.pkl"#
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
            W = data['W']
            bias = data['bias']
        print("Pesos y bias cargados desde el archivo pkl.")
    else:
        print("No se encontró el archivo pkl, favor de entrenar la red.")
        return

    tol_sec = 0.1  # Tolerancia en segundos
    tol_win = int(tol_sec / win_dur)  # Tolerancia en número de ventanas
    costs = []

    for idx, proto in enumerate(W):
        n, m = len(proto), len(test_feats)
        win = max(tol_win, abs(n - m))  # Ventana adaptativa para DTW
        dtw = np.full((n + 1, m + 1), np.inf)  # Inicializar matriz DTW
        dtw[0, 0] = 0

        # Calcular DTW entre el prototipo y las características de prueba
        for j in range(1, n + 1):
            for k in range(max(1, j - win), min(m + 1, j + win)):
                cost = np.linalg.norm(proto[j - 1] - test_feats[k - 1])  # Distancia entre vectores
                dtw[j, k] = cost + min(dtw[j - 1, k], dtw[j, k - 1], dtw[j - 1, k - 1])
        costs.append(dtw[n, m])  # Costo final del emparejamiento

        # Visualización opcional de la matriz DTW
        # plt.figure(figsize=(8, 6))
        # plt.imshow(dtw.T, origin="lower", cmap="hot", interpolation="nearest", aspect="auto")
        # plt.title(f"Matriz DTW para el prototipo {idx+1}")
        # plt.xlabel("Índice del prototipo")
        # plt.ylabel("Índice de la señal de prueba")
        # plt.colorbar(label="Costo")
        # plt.show()
    winner_ind=np.argmin(costs)
    print(f"Combinación: {com[winner_ind]}")

    return  com[winner_ind] # Retorna el índice del prototipo más cercano<<


#Función para el filtrado y manejo de la señal por amplitudes y el ploteo de las señales
def preenfasis_y_graficos(senal, threshold=0.02, alpha=0.95):
    senal_normal = senal / np.max(np.abs(senal))
    if len(senal_normal.shape) > 1:
        senal_normal = senal_normal[:, 0]
    pre = np.roll(senal_normal, 1) - alpha * senal_normal
    pre[0] = 0
    pre = np.where(np.abs(pre) >= 0.9, 0, pre)
    pre_trimmed = pre

    # active_idx = np.where(np.abs(pre) > threshold)[0]
    # if len(active_idx) > 0:
    #     start_idx = active_idx[0]
    #     end_idx = active_idx[-1]
    # #     pre_trimmed = pre[start_idx:end_idx + 1]
    # # else:
    # #     pre_trimmed = pre
    
    # # if len(pre_trimmed) > 100:
    # #     pre_trimmed = pre_trimmed[:100]
    # # elif len(pre_trimmed) < 100:
    # #     pre_trimmed = np.pad(pre_trimmed, (0, 100 - len(pre_trimmed)), mode='constant')

    # duration = len(pre) / tasa_muestreo
    # time = np.linspace(0, duration, len(pre))
    # time_trimmed = np.linspace(start_idx / tasa_muestreo, end_idx / tasa_muestreo, len(pre_trimmed))

    # plt.figure(figsize=(10, 5))
    # plt.subplot(3, 1, 1)
    # plt.plot(time, senal_normal)
    # plt.title('Señal normalizada')
    # plt.xlabel('Tiempo [s]')
    # plt.ylabel('Amplitud')

    # plt.subplot(3, 1, 2)
    # plt.plot(time, pre)
    # plt.title('Señal pre-enfasis')
    # plt.xlabel('Tiempo [s]')
    # plt.ylabel('Amplitud')

    # plt.subplot(3, 1, 3)
    # plt.plot(time_trimmed, pre_trimmed)
    # plt.title('Señal recortada')
    # plt.xlabel('Tiempo [s]')
    # plt.ylabel('Amplitud')

    # plt.tight_layout()
    # # plt.show()
    return pre_trimmed

#Función para el cálculo de energía y cruces por cero
def ener_ZCR(sig,Fs): #Agregar fft
    fft=np.abs(np.fft.fft(sig))
    win_dur = 0.02  
    # Parámetros de la ventana y pasos
    win_len = int(win_dur * Fs)
    step = win_len
    n_windows = int(np.ceil((len(sig) - win_len) / step)) + 1
    # Rellenar señal
    pad_len = n_windows * step + win_len
    sig = np.append(sig, np.zeros(pad_len - len(sig)))

    # Cálculo de energía y ZCR
    eng = []
    zcr = []
    for i in range(n_windows):
        start = i * step
        end = start + win_len
        win = sig[start:end]
        eng.append(np.sum(win ** 2))
        zcr.append(np.sum(np.abs(np.diff(np.sign(win)))) / 2)  
    
    fft=fft/np.max(np.abs(fft))
    eng=eng/np.max(np.abs(eng))
    zcr=zcr/np.max(np.abs(zcr))
    feats = np.concatenate((eng, zcr))
    # feats=feats.flatten()
    # feats = eng
    # feats = zcr
    feats= feats/np.max(np.abs(feats))
    return feats

'''
def ener_ZCR(sig, Fs, freq_min=150, freq_max=300):  
    win_dur = 0.02  
    # Parámetros de la ventana y pasos
    win_len = int(win_dur * Fs)
    step = win_len
    n_windows = int(np.ceil((len(sig) - win_len) / step)) + 1
    # Rellenar señal
    pad_len = n_windows * step + win_len
    sig = np.append(sig, np.zeros(pad_len - len(sig)))

    eng = []
    zcr = []
    fft_concat = []

    freq_max = freq_max if freq_max is not None else Fs / 2 
    f_min_idx = int(freq_min / (Fs / win_len))
    f_max_idx = int(freq_max / (Fs / win_len))

    for i in range(n_windows):
        start = i * step
        end = start + win_len
        win = sig[start:end]
        eng.append(np.sum(win ** 2))
        zcr.append(np.sum(np.abs(np.diff(np.sign(win)))) / 2)  
        fft_win = np.abs(np.fft.fft(win))[f_min_idx:f_max_idx] 
        fft_concat.extend(fft_win)

    # Normalización
    eng = np.array(eng) / np.max(np.abs(eng))
    zcr = np.array(zcr) / np.max(np.abs(zcr))
    fft_concat = np.array(fft_concat) / np.max(np.abs(fft_concat))

    # Concatenar características
    feats = np.concatenate((eng, zcr, fft_concat))
    
    # fft=fft/np.max(np.abs(fft))
    # eng=eng/np.max(np.abs(eng))
    # zcr=zcr/np.max(np.abs(zcr))
    # feats = np.concatenate((eng,zcr,fft))
    # plt.figure()
    # plt.plot(eng)
    # plt.figure()
    # plt.plot(zcr)
    # plt.figure
    # plt.plot(fft_concat)
    # plt.figure
    # plt.plot(feats)
    # feats=feats.flatten()
    # feats = eng
    # feats = zcr
    # feats=fft
    # feats= feats/np.max(np.abs(feats))
    return feats'''

#FUnción para realizar la base de datos
def DB():
    audio_dir = r"D:\Upiita\6to\Patrones\Proyectos\Carrito\audios\train"

    proto_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                proto_files.append(os.path.join(root, file))

    subcarpeta_feats = {}

    for root, dirs, files in os.walk(audio_dir):
        # plt.show()
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                Fs, sig = wav.read(file_path)

                # Normalizar y aplicar preénfasis
                sig = preenfasis_y_graficos(sig)
                feats= ener_ZCR(sig,Fs)
                subcarpeta = os.path.basename(root)  
                
                # Añadir características a la lista correspondiente de la subcarpeta
                if subcarpeta not in subcarpeta_feats:
                    subcarpeta_feats[subcarpeta] = []
                subcarpeta_feats[subcarpeta].append(feats)

    centroides = []

    for subcarpeta, feats_list in subcarpeta_feats.items():
        # # Encontrar la longitud máxima entre los vectores de características
        # max_length = max(len(feats) for feats in feats_list)
        
        # # Rellenar cada vector con ceros al final para igualar longitudes
        # padded_feats = [np.pad(feats, (0, max_length - len(feats)), mode='constant') for feats in feats_list]
        
        # Calcular el centroide (promedio) de las características para esta subcarpeta
        # centroid = np.mean(feats_list, axis=0)
        centroid = feats_list[0]
        
        # Guardar el centroide
        centroides.append(centroid)

    # Convertir los centroides a un arreglo numpy
    W = np.array(centroides, dtype=np.float64)
    # W = np.random.uniform(0, 0.5, size=W.shape)
    bias = np.zeros(W.shape[0])
    pkl = 'pesos_bias.pkl'
    print(W)
    print(W.size)
    # print(f"Prototipos: {proto_feats}")
    print(subcarpeta_feats)
    with open(pkl, 'wb') as f:
        pickle.dump({'W': W, 'bias': bias}, f)
    return W,subcarpeta_feats

#Menú Principal e implementación
arduino = None
while True:
    # plt.show()
    print("\n=== Menú Principal ===")
    print("\nSelecciona una opción:")
    print("0. Vincular Arduino")
    print("1. Grabar audio en tiempo real")
    print("2. Cargar archivo de audio pregrabado")
    print("3. Agregar audio a base de datos")
    print("4. Determinar pesos sinápticos")
    print("5. Entrenar Red")
    print("6. Salir")
    opcion = input("Opción: ")

    if opcion == "0":
        if arduino and arduino.is_open:
            arduino.close()
        try:
            arduino = serial.Serial('COM19', 9600, timeout=2, dsrdtr=True)
            time.sleep(2)
            print("Conexión exitosa.")
        except serial.SerialException as e:
            print(f"Error al conectar con Arduino: {e}")

    elif opcion == "1":
        # print("Grabando audio...")
        nombre_archivo = 'audio_grabado.wav'
        formato = pyaudio.paInt16
        canales = 1
        tasa_muestreo = 44100
        tamano_bloque = 1024
        tiempo_grabacion = 2
        senal = grabar_audio(nombre_archivo)
        if senal is not None:
            sig=preenfasis_y_graficos(senal)

        feats= ener_ZCR(sig,tasa_muestreo)
        res=Compet_Recur_Ts(feats)
        res=DTW(feats)
        print(res)
        sarduino(res)

    elif opcion == "2":
        file_path = input("Introduce la ruta del archivo WAV: ").strip().strip('"')
        if os.path.exists(file_path):
            senal, tasa_muestreo = cargar_audio(file_path)
            sig=preenfasis_y_graficos(senal)
            feats= ener_ZCR(sig,tasa_muestreo)
            res=Compet_Recur_Ts(feats)
            res=DTW(feats) 
            sarduino(res)
        else:
            print("El archivo no existe. Intenta de nuevo.")

    elif opcion == "3":
        while True:
            print("\n=== Menú de Grabación ===")
            print("1. Grabar puntos cardinales")
            # print("1. Grabar inglés")
            print("2. Grabar español")
            print("3. Salir\n")
            opcion = input("Selecciona una opción: ")
            if opcion == "1":
                print("1. Grabar norte")
                print("2. Grabar sur")
                print("3. Grabar este")
                print("4. Grabar oeste")
                # print("1. Grabar forward")
                # print("2. Grabar backward")
                # print("3. Grabar left")
                # print("4. Grabar right")
                print("5. Volver")

                opcion = input("Selecciona una opción: ")

                if opcion in ["1", "2", "3", "4"]:
                    tipo = en[int(opcion) - 1]
                    muestra = int(input(f"Introduce el número de muestra para {tipo}: "))
                    add_db(tipo, muestra)
                elif opcion == "5":
                    print("Volviendo al menú anterior...")
                    continue
                else:
                    print("Opción no válida. Intenta de nuevo.")
            elif opcion=="2":
                print("1. Grabar adelante")
                print("2. Grabar atras")
                print("3. Grabar izquierda")
                print("4. Grabar derecha")
                print("5. Volver")

                opcion = input("Selecciona una opción: ")

                if opcion in ["1", "2", "3", "4"]:
                    tipo = es[int(opcion) - 1]
                    muestra = int(input(f"Introduce el número de muestra para {tipo}: "))
                    add_db(tipo, muestra)
                elif opcion == "5":
                    print("Volviendo al menú anterior...")
                    continue
                else:
                    print("Opción no válida. Intenta de nuevo.")
            elif opcion == "3":
                print("Saliendo al menú principal...")
                break

    elif opcion == "4":
        W,patronesdb=DB()

    elif opcion == "5":
        valores = []
        # print(patronesdb)
        for key, feats_list in patronesdb.items():
            for feats in feats_list:
                valores.append(feats)
        
        # Encuentra la longitud máxima de las características
        # max_length = max(len(feats) for feats in valores)
        
        # # Normaliza las características rellenando con ceros al final
        # valores = [
        #     np.pad(feats, (0, max_length - len(feats)), mode='constant') for feats in valores
        # ]
        
        # Convierte la lista de características a un array NumPy
        patrones = np.array(valores, dtype=np.float64)
        # print(f"Forma de patrones para entrenamiento: {patrones.shape}")
        print(patrones)
        # Compet_Recur_Tr(patrones, W)
        Compet_Recur_Tr(patrones,W)

    elif opcion == "6":
        if arduino and arduino.is_open:
          arduino.close()
        print("Saliendo...")
        break
    else:
        print("Opción no válida. Intenta de nuevo.")

# plt.show()