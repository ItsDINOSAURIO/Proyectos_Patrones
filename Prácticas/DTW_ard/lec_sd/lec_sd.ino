// #include <SPI.h>
// #include <SD.h>

// File dataFile;

// void setup() {
//   Serial.begin(9600);
//   while (!Serial); // Espera a que se abra el puerto serial

//   // Inicializa la tarjeta SD
//   if (!SD.begin(10)) {
//     Serial.println("Error inicializando la tarjeta SD");
//     while (1);
//   }

//   // Procesa el archivo CSV
//   Serial.println("Procesando archivo CSV...");
//   if (processCSV()) {
//     Serial.println("Archivo CSV procesado correctamente.");
//   } else {
//     Serial.println("Error al procesar el archivo CSV.");
//   }
// }

// void loop() {
//   // No hay operaciones en el loop
// }

// /**
//  * Procesa el archivo CSV línea por línea
//  */
// bool processCSV() {
//   dataFile = SD.open("database.csv");
//   if (!dataFile) {
//     Serial.println("Error al abrir el archivo CSV.");
//     return false;
//   }

//   while (dataFile.available()) {
//     String line = dataFile.readStringUntil('\n'); // Lee una línea
//     line.trim(); // Elimina espacios y saltos de línea

//     if (line.length() == 0) {
//       continue; // Ignora líneas vacías
//     }

//     // Divide la línea por comas
//     int firstComma = line.indexOf(',');
//     int lastComma = line.lastIndexOf(',');

//     if (firstComma == -1 || lastComma == -1 || firstComma == lastComma) {
//       Serial.println("Formato incorrecto en la línea:");
//       Serial.println(line);
//       continue;
//     }

//     // Extrae las columnas
//     String energyStr = line.substring(firstComma + 1, lastComma);
//     String zcrStr = line.substring(lastComma + 1);
//     String classStr = line.substring(0, firstComma);

//     // Convierte a números
//     float energy = energyStr.toFloat();
//     float zcr = zcrStr.toFloat();
//     int classLabel = classStr.toInt();

//     // Muestra la salida
//     Serial.print("Clase ");
//     Serial.print(classLabel);
//     Serial.print(": ");
//     Serial.print(energy, 2);
//     Serial.print(" ");
//     Serial.println(zcr);
//   }

//   dataFile.close();
//   return true;
// }


#include <SPI.h>
#include <SD.h>

File dataFile;
int selectedClass = -1; // Clase seleccionada, -1 indica que no se ha seleccionado ninguna

void setup() {
  Serial.begin(9600);
  while (!Serial); // Espera a que se abra el puerto serial

  // Inicializa la tarjeta SD
  if (!SD.begin(10)) {
    Serial.println("Error inicializando la tarjeta SD");
    while (1);
  }

  // Solicita la clase al usuario
  Serial.println("Ingrese la clase deseada (número entero) y presione Enter:");
  while (selectedClass == -1) {
    if (Serial.available() > 0) {
      String input = Serial.readStringUntil('\n');
      selectedClass = input.toInt();
      if (selectedClass >= 0) {
        Serial.print("Clase seleccionada: ");
        Serial.println(selectedClass);
      } else {
        Serial.println("Entrada no válida. Ingrese un número entero no negativo:");
      }
    }
  }

  // Procesa el archivo CSV
  Serial.println("Procesando archivo CSV...");
  if (processCSV()) {
    Serial.println("Archivo CSV procesado correctamente.");
  } else {
    Serial.println("Error al procesar el archivo CSV.");
  }
}

void loop() {
  // No hay operaciones en el loop
}

/**
 * Procesa el archivo CSV línea por línea
 */
// bool processCSV() {
//   dataFile = SD.open("database.csv");
//   if (!dataFile) {
//     Serial.println("Error al abrir el archivo CSV.");
//     return false;
//   }

//   while (dataFile.available()) {
//     String line = dataFile.readStringUntil('\n'); // Lee una línea
//     line.trim(); // Elimina espacios y saltos de línea

//     if (line.length() == 0) {
//       continue; // Ignora líneas vacías
//     }

//     // Divide la línea por comas
//     int firstComma = line.indexOf(',');
//     int lastComma = line.lastIndexOf(',');

//     if (firstComma == -1 || lastComma == -1 || firstComma == lastComma) {
//       Serial.println("Formato incorrecto en la línea:");
//       Serial.println(line);
//       continue;
//     }

//     // Extrae las columnas
//     String energyStr = line.substring(firstComma + 1, lastComma);
//     String zcrStr = line.substring(lastComma + 1);
//     String classStr = line.substring(0, firstComma);

//     // Convierte a números
//     float energy = energyStr.toFloat();
//     float zcr = zcrStr.toFloat();
//     int classLabel = classStr.toInt();

//     // Filtra por la clase seleccionada
//     if (classLabel == selectedClass) {
//       Serial.print("Clase ");
//       Serial.print(classLabel);
//       Serial.print(": ");
//       Serial.print(energy, 10);
//       Serial.print(" ");
//       Serial.println(zcr);
//     }
//   }

//   dataFile.close();
//   return true;
// }

float referencePoint[2] = {3.0, 4.0}; // Punto de referencia (x, y)

bool processCSV() {
  dataFile = SD.open("database.csv");
  if (!dataFile) {
    Serial.println("Error al abrir el archivo CSV.");
    return false;
  }

  for (int selectedClass = 0; selectedClass < 10; selectedClass++) { // Itera entre clases 0 a 9
    Serial.print("Procesando clase ");
    Serial.println(selectedClass);

    dataFile.seek(0); // Reinicia la lectura del archivo desde el principio

    while (dataFile.available()) {
      String line = dataFile.readStringUntil('\n'); // Lee una línea
      line.trim(); // Elimina espacios y saltos de línea

      if (line.length() == 0) {
        continue; // Ignora líneas vacías
      }

      // Divide la línea por comas
      int firstComma = line.indexOf(',');
      int lastComma = line.lastIndexOf(',');
            Serial.println(line);
      Serial.println(firstComma);
      Serial.println(lastComma);

      if (firstComma == -1 || lastComma == -1 || firstComma == lastComma) {
        Serial.println("Formato incorrecto en la línea:");
        Serial.println(line);
        continue;
      }

      // Extrae las columnas
      String energyStr = line.substring(firstComma + 1, lastComma);
      String zcrStr = line.substring(lastComma + 1);
      String classStr = line.substring(0, firstComma);

      // Convierte a números
      float energy = energyStr.toFloat();
      float zcr = zcrStr.toFloat();
      int classLabel = classStr.toInt();

      // Filtra por la clase actual en la iteración
      if (classLabel == selectedClass) {
        // Calcula la distancia euclidiana al punto de referencia
        float distance = sqrt(pow((referencePoint[0] - energy), 2) +
                              pow((referencePoint[1] - zcr), 2));

        // Imprime los resultados
        Serial.print("Clase ");
        Serial.print(classLabel);
        Serial.print(": ");
        Serial.print("Energy: ");
        Serial.print(energy, 2);
        Serial.print(", ZCR: ");
        Serial.print(zcr, 2);
        Serial.print(", Distancia al punto de referencia: ");
        Serial.println(distance, 2);
      }
    }
  }

  dataFile.close();
  return true;
}
