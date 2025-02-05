//Librerías
#include <SPI.h>
#include <SD.h>
#include <SoftwareSerial.h>

// Configuración para Serial1 en pines 8 (Rx) y 9 (Tx)
// SoftwareSerial bluetooth(8, 9);

// Vars globales
int step = 0;
int win_len = 0;
int n_windows = 1;

// Vars para los cálculos con memoria dinámica
float window[100];
double* energy = nullptr;
double* zcr = nullptr;
double** test_feats = nullptr;
double* costs = nullptr;
//Parámetros de ventana
int window_idx = 0;
int window_C = 0;

bool params_received = false;
int selectedClass = -1;

// Función que al recibir parámetros indica el comienzo del código
void recibirParametros() {
  if (Serial1.available() > 0) {
    String input = Serial1.readStringUntil('\n');
    input.trim();

    int comma1 = input.indexOf(',');
    int comma2 = input.lastIndexOf(',');
    // Serial.println(comma1);
    // Serial.println(comma2);

    if (comma1 > 0 && comma2 > comma1) {
      step = input.substring(0, comma1).toInt();
      win_len = input.substring(comma1 + 1, comma2).toInt();
      n_windows = input.substring(comma2 + 1).toInt();

      energy = (double*)malloc(n_windows * sizeof(double));
      zcr = (double*)malloc(n_windows * sizeof(double));
      test_feats = (double**)malloc(n_windows * sizeof(double*));
      costs = (double*)malloc(n_windows * sizeof(double));

      if (energy && zcr && test_feats && costs) {
        for (int i = 0; i < n_windows; i++) {
          test_feats[i] = (double*)malloc(2 * sizeof(double));
          if (!test_feats[i]) {
            Serial.println("Error: Memoria insuficiente.");
            Serial1.println("0");
            return;
          }
        }

        params_received = true;
        Serial.println("Parámetros recibidos correctamente.");
        Serial.print("step: "); Serial.print(step);
        Serial.print(", win_len: "); Serial.print(win_len);
        Serial.print(", n_windows: "); Serial.println(n_windows);
        Serial1.println("1");
      } else {
        Serial.println("Error: Memoria insuficiente.");
        Serial1.println("0");
      }
    } else {
      Serial.println("Error: Formato de parámetros inválido.");
    }
  }
}

void procesarDatos() {
  // if (Serial1.available() > 0) {
    Serial1.println("3");
    Serial.println("Esperando datos...");

    String input = Serial1.readStringUntil('\n');
    double sample = input.toDouble();
    Serial.println(sample);

    if (input == "2") {
      Serial.println("Señal recibida, Comenzando Cálculos...")
      gTest();
      aCSV();
      liberarMemoria();
      params_received = false;
      Serial.println("Fin del procesamiento...")
      return;
    }

    if (window_idx < step) {
      window[window_idx++] = sample;
    }

    if (window_idx == step) {
      Serial1.println("4");
      Serial.println("Procesando ventana...");

      double c_energy = 0.0, c_zcr = 0.0;

      for (int i = 0; i < step; i++) {
        c_energy += window[i] * window[i];
      }

      for (int i = 1; i < step; i++) {
        if ((window[i - 1] > 0 && window[i] < 0) || (window[i - 1] < 0 && window[i] > 0)) {
          c_zcr++;
        }
      }

      if (window_C < n_windows) {
        energy[window_C] = c_energy;
        zcr[window_C] = c_zcr;
        window_C++;
      }
      window_idx = 0;
    }
  }
}

//Función para guardar los datos recopilados del test
void gTest() {
  for (int i = 0; i < window_C; i++) {
    test_feats[i][0] = energy[i];
    test_feats[i][1] = zcr[i];
  }
}

void aCSV() {
  Serial.println("Analizando base de datos");
  File DB = SD.open("database.csv");//("database.csv")
  if (!DB) {
    Serial.println("Error al abrir 'database.csv'.");
    return;
  }

  const int TOL_WIN = 5;
  double min_costoT = INFINITY;
  int Clase_det = -1;

  double** dtw = (double**)malloc((step + 1) * sizeof(double*));
  for (int i = 0; i <= step; i++) {
    dtw[i] = (double*)malloc((step + 1) * sizeof(double));
  }

  for (int clase = 0; clase < 10; clase++) {
    DB.seek(0);

    while (DB.available()) {
      String linea = DB.readStringUntil('\n');
      linea.trim();

      int coma1 = linea.indexOf(',');
      int coma2 = linea.lastIndexOf(',');
      if (coma1 == -1 || coma2 == -1) continue;
       if (linea.substring(coma1 + 1, coma2)=="Energy" || linea.substring(coma2 + 1)=="ZCR"||linea.substring(0, coma1)=="Class"){
        continue;
      } else{

      double csv_energy = linea.substring(coma1 + 1, coma2).toDouble();
      double csv_zcr = linea.substring(coma2 + 1).toDouble();
      int classLabel = linea.substring(0, coma1).toInt();

      if (classLabel == clase) {
        for (int idx = 0; idx < window_C; idx++) {
          for (int i = 0; i <= step; i++) {
            for (int j = 0; j <= step; j++) {
              dtw[i][j] = (i == 0 && j == 0) ? 0 : INFINITY;
            }
          }

          for (int i = 1; i <= step; i++) {
            for (int j = max(1, i - TOL_WIN); j <= min(step, i + TOL_WIN); j++) {
              double cost = sqrt(pow(csv_energy - test_feats[idx][0], 2) + pow(csv_zcr - test_feats[idx][1], 2));
              dtw[i][j] = cost + min(min(dtw[i - 1][j], dtw[i][j - 1]), dtw[i - 1][j - 1]);

            }
          }

          double final_cost = dtw[step][step];
          if (final_cost < costs[idx]) costs[idx] = final_cost;
        }
      }
    }
  }
  }
  for (int i = 0; i <= step; i++) {
    free(dtw[i]);
  }
  free(dtw);

  DB.close();

  double total_cost = 0;
  for (int i = 0; i < window_C; i++) {
    total_cost += costs[i];
  }

  if (total_cost < min_costoT) {
    min_costoT = total_cost;
    Clase_det = selectedClass;
  }

  Serial.print("El número es: ");
  Serial.println(Clase_det);
  Serial1.print(Clase_det);
}

//Función para liberar memoria dinámica
void liberarMemoria() {
  free(energy);
  free(zcr);
  free(costs);
  for (int i = 0; i < n_windows; i++) {
    free(test_feats[i]);
  }
  free(test_feats);
}

void setup() {
  Serial.begin(9600);
  Serial1.begin(115200);
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