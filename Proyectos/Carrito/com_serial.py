# import serial, time
# arduino = serial.Serial('COM4', 9600)
# time.sleep(1)
# rawString = arduino.readline()
# while True:
#     print(rawString)
# # arduino.close()

# import serial
# import time

# # Configuración del puerto serial
# arduino = serial.Serial('COM4', 9600)
# time.sleep(2)  # Espera para asegurar que la conexión esté establecida

# # Ejemplo de condición para enviar datos al Arduino
# condicion = True  # Cambia esta condición según tus necesidades

# if condicion:
#     # Enviar datos al Arduino
#     mensaje = "Mensaje desde Python"
#     arduino.write(mensaje.encode())  # Convierte el mensaje a bytes y envía

# # Leer respuesta del Arduino (si la tiene)
# while arduino.in_waiting > 0:
#     rawString = arduino.readline().decode('utf-8').strip()
#     print(rawString)

# arduino.close()

# import serial
# import time

# arduino = serial.Serial('COM4', 9600)
# time.sleep(2)  # Espera para que el Arduino se reinicie y configure el puerto correctamente

# # Enviar un mensaje al Arduino
# arduino.write("Hola Arduino".encode())

# # Intentar leer respuesta
# time.sleep(1)  # Espera un momento antes de leer para asegurarse de que haya datos
# while arduino.in_waiting > 0:
#     rawString = arduino.readline().decode('utf-8').strip()
#     print("Respuesta del Arduino:", rawString)

# arduino.close()

# import serial
# import time

# arduino = serial.Serial('COM4', 9600)
# time.sleep(2)  # Espera para que el Arduino se reinicie

# # Enviar un mensaje al Arduino
# mensaje = "Hola Arduino"
# arduino.write(mensaje.encode())

# # Leer la respuesta del Arduino
# time.sleep(1)  # Espera para asegurarse de que haya tiempo de respuesta
# while arduino.in_waiting > 0:
#     respuesta = arduino.readline().decode('utf-8').strip()
#     print("Respuesta del Arduino:", respuesta)

# arduino.close()

# import serial
# import time

# # Configura el puerto COM y velocidad (baudrate)
# bluetooth_port = "COM16"  # Cambia por el puerto asignado a tu Bluetooth
# baudrate = 115200

# # Inicia la conexión serial
# try:
#     bluetooth = serial.Serial(bluetooth_port, baudrate)
#     print("Conexión Bluetooth establecida.")

#     # Enviar datos al módulo Bluetooth
#     while True:
#         message = input("Escribe un mensaje para enviar: ")
#         bluetooth.write(message.encode())  # Envía datos como bytes
#         print("Mensaje enviado:", message)

#         # Leer respuesta del Bluetooth (si Arduino envía algo)
#         if bluetooth.in_waiting > 0:
#             response = bluetooth.readline().decode().strip()
#             print("Respuesta recibida:", response)
            
# except serial.SerialException as e:
#     print(f"No se pudo conectar al puerto {bluetooth_port}. Verifica la conexión.")
# finally:
#     if 'bluetooth' in locals():
#         bluetooth.close()
#         print("Conexión Bluetooth cerrada.")


import serial
import time

# Configura el puerto COM y velocidad (baudrate)
bluetooth_port = "COM16"  # Cambia por el puerto asignado a tu Bluetooth
baudrate = 115200

# Inicia la conexión serial
try:
    bluetooth = serial.Serial(bluetooth_port, baudrate, timeout=1)  # Timeout agregado para evitar bloqueos
    print("Conexión Bluetooth establecida.")

    # Enviar datos al módulo Bluetooth
    while True:
        message = input("Escribe un mensaje para enviar ('DISCONNECT' para terminar): ")
        bluetooth.write(message.encode() + b'\n')  # Envía datos con un salto de línea
        print("Mensaje enviado:", message)

        # Leer respuesta del Bluetooth (si Arduino envía algo)
        if bluetooth.in_waiting > 0:
            response = bluetooth.readline().decode().strip()
            print("Respuesta recibida:", response)

        # Finalizar conexión si se envía el comando "DISCONNECT"
        if message == "DISCONNECT":
            print("Cerrando conexión Bluetooth...")
            bluetooth.close()
            print("Conexión Bluetooth cerrada.")
            break

except serial.SerialException as e:
    print(f"No se pudo conectar al puerto {bluetooth_port}. Verifica la conexión. Error: {e}")
except Exception as e:
    print(f"Ocurrió un error: {e}")
finally:
    # Asegura que la conexión se cierre si ocurre un error o el programa termina
    if 'bluetooth' in locals() and bluetooth.is_open:
        bluetooth.close()
        print("Conexión Bluetooth cerrada.")
