import wave
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import os
import serial
import time
from scipy import signal
import pickle
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
en = ["forward", "backward", "left", "right"]
es = ["adelante", "atras", "izquierda", "derecha"]
# com = ("izquierda", "adelante", "backward", "derecha", "forward", "atras","left", "right") 
com = ("11", "00", "10", "01") 
# com = ("izquierda", "adelante", "11", "derecha", "00", "atras","10", "01") 
 #00 adelante, 10 izquierda, 01 derecha, 11 atras

#FUnción para agregar nuevos audios a la base de datos
def add_db(tipo, muestra):
    directorio_base = r"D:\Upiita\6to\Patrones\Proyectos\Carrito\audios\train"
    if not os.path.exists(directorio_base):
        os.makedirs(directorio_base)

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
def Instar_Compet_Tr(patrones, W):
    pkl = 'pesos_bias.pkl'
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
            W = data['W']
            bias = data['bias']
        print("Pesos y bias cargados desde el archivo pkl.")
    else:
        print("No se encontró el archivo pkl, se utilizarán los pesos iniciales.")
        bias = np.ones(W.shape[0])

    epocas = 10000
    alpha = 0.0001

    for _ in range(epocas):
        for i in range(patrones.shape[0]):
            a1 = np.dot(W, patrones[i, :].T) + bias
            ganador = np.argmax(a1)
            for j in range(W.shape[0]):
                if j == ganador:
                    W[j, :] += alpha * (patrones[i, :] - W[j, :])
            bias[ganador] -= 0.2 * (1 + bias[ganador])
    
        with open(pkl, 'wb') as f:
            pickle.dump({'W': W, 'bias': bias}, f)

#Función para probar la red
def Instar_Compet_Ts(patron, W):
    pkl = 'pesos_bias.pkl'
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
            W = data['W']
            bias = data['bias']
        print("Pesos y bias cargados desde el archivo pkl.")
    else:
        print("No se encontró el archivo pkl, favor de entrenar la red.")
        return

    epsilon = 0.05 
    iteracion = 300

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
    print(f"Secuencia: {com[winner_ind]}")
    
    return com[winner_ind]

#Función para el filtrado y manejo de la señal por amplitudes y el ploteo de las señales
def preenfasis_y_graficos(senal, threshold=0.1, alpha=0.95):
    senal_normal = senal / np.max(np.abs(senal))
    if len(senal_normal.shape) > 1:
        senal_normal = senal_normal[:, 0]
    pre = np.roll(senal_normal, 1) - alpha * senal_normal
    pre[0] = 0
    pre = np.where(np.abs(pre) >= 0.9, 0, pre)

    # active_idx = np.where(np.abs(pre) > threshold)[0]
    # if len(active_idx) > 0:
    #     start_idx = active_idx[0]
    #     end_idx = active_idx[-1]
    #     pre_trimmed = pre[start_idx:end_idx + 1]
    # else:
    pre_trimmed = pre

    # if len(pre_trimmed) > 100:
    #     pre_trimmed = pre_trimmed[:100]
    # elif len(pre_trimmed) < 100:
    #     pre_trimmed = np.pad(pre_trimmed, (0, 100 - len(pre_trimmed)), mode='constant')

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
    # plt.show()
    return pre_trimmed

#Función para el cálculo de energía y cruces por cero
def ener_ZCR(n_windows,step,win_len,sig_pad):
    eng = []
    zcr = []
    for i in range(n_windows):
        start = i * step
        end = start + win_len
        win = sig_pad[start:end]
        eng.append(np.sum(win ** 2))  
        zcr.append(np.sum(np.abs(np.diff(np.sign(win)))) / 2)  
    
    feats = np.concatenate((eng, zcr))
    return feats

#Función para realizar la base de datos
def DB():
    audio_dir = r"D:\Upiita\6to\Patrones\Proyectos\Carrito\audios\train"
    proto_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                proto_files.append(os.path.join(root, file))

    subcarpeta_feats = {}
    win_dur = 0.02  

    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                Fs, sig = wav.read(file_path)

                sig = preenfasis_y_graficos(sig)
                
                win_len = int(win_dur * Fs)
                step = win_len
                n_windows = int(np.ceil((len(sig) - win_len) / step)) + 1
                
                pad_len = n_windows * step + win_len
                sig_pad = np.append(sig, np.zeros(pad_len - len(sig)))

                feats= ener_ZCR(n_windows,step,win_len,sig_pad)
                subcarpeta = os.path.basename(root)
                
                if subcarpeta not in subcarpeta_feats:
                    subcarpeta_feats[subcarpeta] = []
                subcarpeta_feats[subcarpeta].append(feats)

    centroides = []

    for subcarpeta, feats_list in subcarpeta_feats.items():
        # max_length = max(len(feats) for feats in feats_list)
        
        # padded_feats = [np.pad(feats, (0, max_length - len(feats)), mode='constant') for feats in feats_list]
        
        centroid = np.mean(feats, axis=0)
        
        centroides.append(centroid)

    # Convertir los centroides a un arreglo numpy
    W = np.array(centroides, dtype=np.float64)
    print(W)
    print(W.size)
    # print(f"Prototipos: {proto_feats}")
    print(subcarpeta_feats)
    return W,subcarpeta_feats

#Menú Principal e implementación
arduino = None
while True:
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
            arduino = serial.Serial('COM16', 115200, timeout=2, dsrdtr=True)
            time.sleep(2)
            print("Conexión exitosa.")
        except serial.SerialException as e:
            print(f"Error al conectar con Arduino: {e}")

    elif opcion == "1":
        print("Grabando audio...")
        nombre_archivo = 'audio_grabado.wav'
        formato = pyaudio.paInt16
        canales = 1
        tasa_muestreo = 44100
        tamano_bloque = 1024
        tiempo_grabacion = 2
        senal = grabar_audio(nombre_archivo)
        if senal is not None:
            sig=preenfasis_y_graficos(senal)

        win_dur = 0.02
        # Parámetros de la ventana y pasos
        win_len = int(win_dur * tasa_muestreo)
        step = win_len
        n_windows = int(np.ceil((len(sig) - win_len) / step)) + 1
        
        # Rellenar señal
        pad_len = n_windows * step + win_len
        sig_pad = np.append(sig, np.zeros(pad_len - len(sig)))

        feats= ener_ZCR(n_windows,step,win_len,sig_pad)
        res=Instar_Compet_Ts(feats,W)
        print(res)
        sarduino(res)

    elif opcion == "2":
        file_path = input("Introduce la ruta del archivo WAV: ").strip().strip('"')
        if os.path.exists(file_path):
            senal, tasa_muestreo = cargar_audio(file_path)
            sig=preenfasis_y_graficos(senal)
            win_dur = 0.02
            # Parámetros de la ventana y pasos
            win_len = int(win_dur * tasa_muestreo)
            step = win_len
            n_windows = int(np.ceil((len(sig) - win_len) / step)) + 1
            
            # Rellenar señal
            pad_len = n_windows * step + win_len
            sig_pad = np.append(sig, np.zeros(pad_len - len(sig)))

            feats= ener_ZCR(n_windows,step,win_len,sig_pad)
            res=Instar_Compet_Ts(feats,W)
            sarduino(res)
        else:
            print("El archivo no existe. Intenta de nuevo.")

    elif opcion == "3":
        while True:
            print("\n=== Menú de Grabación ===")
            print("1. Grabar inglés")
            print("2. Grabar español")
            print("3. Salir\n")
            opcion = input("Selecciona una opción: ")
            if opcion == "1":
                print("1. Grabar forward")
                print("2. Grabar backward")
                print("3. Grabar left")
                print("4. Grabar right")
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
            # Obtén las características desde subcarpeta_feats
        valores = []
        print(patronesdb)
        for key, feats_list in patronesdb.items():
            for feats in feats_list:
                valores.append(feats)  # Asegúrate de agregar todos los arrays a una lista plana
        
        # Encuentra la longitud máxima de las características
        max_length = max(len(feats) for feats in valores)
        
        # Normaliza las características rellenando con ceros al final
        valores_normalizados = [
            np.pad(feats, (0, max_length - len(feats)), mode='constant') for feats in valores
        ]
        
        # Convierte la lista de características a un array NumPy
        patrones = np.array(valores_normalizados, dtype=np.float64)
        # print(f"Forma de patrones para entrenamiento: {patrones.shape}")
        print(patrones)
        # Llama a la función de entrenamiento
        # Instar_Compet_Tr(patrones, W)
        Instar_Compet_Tr(patrones,W)

    elif opcion == "6":
        if arduino and arduino.is_open:
            arduino.close()
        print("Saliendo...")
        break
    else:
        print("Opción no válida. Intenta de nuevo.")