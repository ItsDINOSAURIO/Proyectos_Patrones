#include <SPI.h>
#include <SD.h>
#include <SoftwareSerial.h>

// Bluetooth configuration on pins 8 (Rx) and 9 (Tx)
SoftwareSerial bluetooth(8, 9);

// Global variables
int step = 0;
int win_len = 0;
int n_windows = 1;

// File names for storing different data
const char* WINDOW_FILE = "window.txt";
const char* ENERGY_FILE = "energy.txt";
const char* ZCR_FILE = "zcr.txt";
const char* TEST_FEATS_FILE = "testfeats.txt";
const char* COSTS_FILE = "costs.txt";

// Global variables for tracking
int window_idx = 0;
int window_C = 0;

bool params_received = false;
int selectedClass = -1;

// Function to write data to a file
void writeToFile(const char* filename, double value) {
  File dataFile = SD.open(filename, FILE_WRITE);
  if (dataFile) {
    dataFile.println(value);
    dataFile.close();
  } else {
    Serial.print("Error opening "); 
    Serial.println(filename);
  }
}

// Function to write data to a file with an index
void writeToFileIndexed(const char* filename, double value, int index) {
  File dataFile = SD.open(filename, FILE_WRITE);
  if (dataFile) {
    dataFile.print(index);
    dataFile.print(",");
    dataFile.println(value);
    dataFile.close();
  } else {
    Serial.print("Error opening "); 
    Serial.println(filename);
  }
}

// Function to read a specific line from a file
double readFromFileIndexed(const char* filename, int index) {
  File dataFile = SD.open(filename, FILE_READ);
  if (dataFile) {
    String line;
    while (dataFile.available()) {
      line = dataFile.readStringUntil('\n');
      int comma = line.indexOf(',');
      if (comma != -1) {
        int storedIndex = line.substring(0, comma).toInt();
        if (storedIndex == index) {
          double value = line.substring(comma + 1).toDouble();
          dataFile.close();
          return value;
        }
      }
    }
    dataFile.close();
  }
  return 0.0;
}

// Function to clear a file
void clearFile(const char* filename) {
  SD.remove(filename);
  File dataFile = SD.open(filename, FILE_WRITE);
  if (dataFile) {
    dataFile.close();
  }
}

void recibirParametros() {
  if (bluetooth.available() > 0) {
    String input = bluetooth.readStringUntil('\n');
    input.trim();

    int comma1 = input.indexOf(',');
    int comma2 = input.lastIndexOf(',');

    if (comma1 > 0 && comma2 > comma1) {
      step = input.substring(0, comma1).toInt();
      win_len = input.substring(comma1 + 1, comma2).toInt();
      n_windows = input.substring(comma2 + 1).toInt();

      // Clear previous data files
      clearFile(WINDOW_FILE);
      clearFile(ENERGY_FILE);
      clearFile(ZCR_FILE);
      clearFile(TEST_FEATS_FILE);
      clearFile(COSTS_FILE);

      params_received = true;
      Serial.println("Parameters received correctly.");
      Serial.print("step: "); Serial.print(step);
      Serial.print(", win_len: "); Serial.print(win_len);
      Serial.print(", n_windows: "); Serial.println(n_windows);
      bluetooth.println("1");
    } else {
      Serial.println("Error: Invalid parameter format.");
      bluetooth.println("0");
    }
  }
}

void procesarDatos() {
  if (bluetooth.available() > 0) {
    String input = bluetooth.readStringUntil('\n');
    if (input == "000") {
      storeFeatures();
      processCSV();
      params_received = false;
      return;
    }

    // Convertir la entrada a double
    double sample = input.toDouble();
    Serial.println(sample);

    // Almacenar datos de la ventana
    if (window_idx < step) {
      bluetooth.println("READY");
      writeToFileIndexed(WINDOW_FILE, sample, window_idx);
      window_idx++;
    }

    // Procesar la ventana si está completa
    if (window_idx == step) {
      bluetooth.println("WAIT");  // Pausar el envío desde Python durante el procesamiento

      double c_energy = 0.0, c_zcr = 0.0;

      // Calcular energía
      for (int i = 0; i < step; i++) {
        double windowSample = readFromFileIndexed(WINDOW_FILE, i);
        c_energy += windowSample * windowSample;
      }

      // Calcular ZCR
      for (int i = 1; i < step; i++) {
        double prevSample = readFromFileIndexed(WINDOW_FILE, i - 1);
        double currentSample = readFromFileIndexed(WINDOW_FILE, i);
        if ((prevSample > 0 && currentSample < 0) || (prevSample < 0 && currentSample > 0)) {
          c_zcr++;
        }
      }

      // Guardar resultados en archivos
      if (window_C < n_windows) {
        writeToFileIndexed(ENERGY_FILE, c_energy, window_C);
        writeToFileIndexed(ZCR_FILE, c_zcr, window_C);
        window_C++;
      }

      // Reiniciar índice de ventana
      window_idx = 0;

      bluetooth.println("READY"); // Reanudar el envío desde Python tras el procesamiento
    }
  }
}

void storeFeatures() {
  for (int i = 0; i < window_C; i++) {
    double energy = readFromFileIndexed(ENERGY_FILE, i);
    double zcr = readFromFileIndexed(ZCR_FILE, i);
    
    // Store test features to file
    File testFeatsFile = SD.open(TEST_FEATS_FILE, FILE_WRITE);
    if (testFeatsFile) {
      testFeatsFile.print(i);
      testFeatsFile.print(",");
      testFeatsFile.print(energy);
      testFeatsFile.print(",");
      testFeatsFile.println(zcr);
      testFeatsFile.close();
    }
  }
}

void processCSV() {
  File DB = SD.open("database.csv");
  if (!DB) {
    Serial.println("Error opening 'database.csv'.");
    return;
  }

  const int TOL_WIN = 5;
  double min_total_cost = INFINITY;
  int recognized_class = -1;

  // Use a file to track minimum costs
  clearFile(COSTS_FILE);

  for (int classID = 0; classID < 10; classID++) {
    DB.seek(0);

    while (DB.available()) {
      String line = DB.readStringUntil('\n');
      line.trim();

      int firstComma = line.indexOf(',');
      int lastComma = line.lastIndexOf(',');
      if (firstComma == -1 || lastComma == -1) continue;

      double csv_energy = line.substring(firstComma + 1, lastComma).toDouble();
      double csv_zcr = line.substring(lastComma + 1).toDouble();
      int classLabel = line.substring(0, firstComma).toInt();

      if (classLabel == classID) {
        for (int idx = 0; idx < window_C; idx++) {
          // Create a temporary DTW file to store calculations
          File dtwFile = SD.open("dtw_temp.txt", FILE_WRITE);
          
          // Initialize DTW with maximum values
          for (int i = 0; i <= step; i++) {
            for (int j = 0; j <= step; j++) {
              dtwFile.print(i);
              dtwFile.print(",");
              dtwFile.print(j);
              dtwFile.print(",");
              dtwFile.println((i == 0 && j == 0) ? 0 : INFINITY);
            }
          }
          dtwFile.close();

          // Perform DTW calculations
          double final_cost = performDTWCalculation(csv_energy, csv_zcr, idx);
          
          // Update costs file
          writeToFileIndexed(COSTS_FILE, final_cost, idx);
        }
      }
    }
  }

  DB.close();

  // Calculate total cost
  double total_cost = 0;
  for (int i = 0; i < window_C; i++) {
    total_cost += readFromFileIndexed(COSTS_FILE, i);
  }

  if (total_cost < min_total_cost) {
    min_total_cost = total_cost;
    recognized_class = selectedClass;
  }

  Serial.print("The number is: ");
  Serial.println(recognized_class);
  bluetooth.print(recognized_class);
}

double performDTWCalculation(double csv_energy, double csv_zcr, int idx) {
  // Read test features for the specific index
  double test_energy = readFromFileIndexed(TEST_FEATS_FILE, idx);
  double test_zcr = readFromFileIndexed(TEST_FEATS_FILE, idx + 1);

  const int TOL_WIN = 5;
  double min_cost = INFINITY;

  File dtwFile = SD.open("dtw_temp.txt", FILE_READ);
  if (!dtwFile) {
    Serial.println("Error opening DTW file");
    return INFINITY;
  }

  // DTW calculation logic similar to previous implementation
  // But now reading and writing to files instead of using dynamic memory
  while (dtwFile.available()) {
    String line = dtwFile.readStringUntil('\n');
    int firstComma = line.indexOf(',');
    int secondComma = line.indexOf(',', firstComma + 1);
    
    int i = line.substring(0, firstComma).toInt();
    int j = line.substring(firstComma + 1, secondComma).toInt();
    double current_dtw = line.substring(secondComma + 1).toDouble();

    if (i > 0 && j > 0) {
      double cost = sqrt(pow(csv_energy - test_energy, 2) + pow(csv_zcr - test_zcr, 2));
      double min_prev_dtw = min(min(
        readDTWValue("dtw_temp.txt", i-1, j), 
        readDTWValue("dtw_temp.txt", i, j-1)
      ), readDTWValue("dtw_temp.txt", i-1, j-1));

      double new_dtw_value = cost + min_prev_dtw;
      writeDTWValue("dtw_temp.txt", i, j, new_dtw_value);

      if (i == step && j == step) {
        min_cost = new_dtw_value;
      }
    }
  }

  dtwFile.close();
  return min_cost;
}

// Helper functions to read and write DTW values to file
double readDTWValue(const char* filename, int i, int j) {
  File dtwFile = SD.open(filename, FILE_READ);
  if (dtwFile) {
    while (dtwFile.available()) {
      String line = dtwFile.readStringUntil('\n');
      int firstComma = line.indexOf(',');
      int secondComma = line.indexOf(',', firstComma + 1);
      
      int fileI = line.substring(0, firstComma).toInt();
      int fileJ = line.substring(firstComma + 1, secondComma).toInt();
      
      if (fileI == i && fileJ == j) {
        double value = line.substring(secondComma + 1).toDouble();
        dtwFile.close();
        return value;
      }
    }
    dtwFile.close();
  }
  return INFINITY;
}

void writeDTWValue(const char* filename, int i, int j, double value) {
  File dtwFile = SD.open(filename, FILE_WRITE);
  if (dtwFile) {
    dtwFile.print(i);
    dtwFile.print(",");
    dtwFile.print(j);
    dtwFile.print(",");
    dtwFile.println(value);
    dtwFile.close();
  }
}

// void setup() {
//   Serial.begin(9600);
//   bluetooth.begin(115200);
//   delay(1000);
//   if (!SD.begin(10)) {
//     Serial.println("Error initializing SD card.");
//     while (1);
//   }
//   Serial.println("System started.");
// }

// void loop() {
//   if (!params_received) {
//     recibirParametros();
//   } else {
//     procesarDatos();
//   }
// }

void setup() {
  Serial.begin(9600);
  bluetooth.begin(115200);
  delay(1000);
  Serial.println(SD.begin(10)); 
  if (!SD.begin(10)) {
    Serial.println("Error inicializando la tarjeta SD.");
    while (1);
  }
  Serial.println("Sistema iniciado.");
}

void loop() {
  if (!params_received) {
    recibirParametros();
  } else {
    procesarDatos();
  }
}