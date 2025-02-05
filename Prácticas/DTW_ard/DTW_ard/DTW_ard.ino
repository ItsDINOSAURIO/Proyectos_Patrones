//Librerías
#include <SPI.h>
#include <SD.h>
#include <SoftwareSerial.h>

//Puerto Serialñ Virtual para Serial1
// SoftwareSerial Serial1(8, 9);
// string bluetooth=Serial1;

//Vars Globales
int step = 0;
int win_len = 0;
int n_windows = 1;

//Archivos
const char* WINDOW_FILE = "window.txt";
const char* ENERGY_FILE = "energy.txt";
const char* ZCR_FILE = "zcr.txt";
const char* TEST_FEATS_FILE = "testfeats.txt";
const char* COSTS_FILE = "costs.txt";
//Parámetros de Ventaneo
int window_idx = 0;
int window_C = 0;

bool param = false;
int result = -1;

//FUnción para la edición de los txt
void editxt(const char* txt, double val, int index) {
  File archivo = SD.open(txt, FILE_WRITE);
  if (archivo) {
    archivo.print(index);
    archivo.print(",");
    archivo.println(val);
    archivo.close();
  } else {
    Serial.print("Error abriendo "); 
    Serial.println(txt);
  }
}
//FUnción para la extracción de datos de los txt
double vertxt(const char* txt, int index) {
  File archivo = SD.open(txt, FILE_READ);
  if (archivo) {
    String linea;
    while (archivo.available()) {
      linea = archivo.readStringUntil('\n');
      int coma = linea.indexOf(',');
      if (coma != -1) {
        int indice = linea.substring(0, coma).toInt();
        if (indice == index) {
          double val = linea.substring(coma + 1).toDouble();
          archivo.close();
          return val;
        }
      }
    }
    archivo.close();
  }
  return 0.0;
}

// Función para reiniciar los txt en caso de que tuviesen información
void rstxt(const char* txt) {
  SD.remove(txt);
  File archivo = SD.open(txt, FILE_WRITE);
  if (archivo) {
    archivo.close();
  }
}

// FUnción que al recibir parámetros indica el comienzo del código
void recibirParametros() {
  if (Serial1.available() > 0) {//Serial.println(Serial1.available());
    String input = Serial1.readStringUntil('\n');
    input.trim();

    int coma1 = input.indexOf(',');
    int coma2 = input.lastIndexOf(',');

    if (coma1 > 0 && coma2 > coma1) {
      step = input.substring(0, coma1).toInt();
      win_len = input.substring(coma1 + 1, coma2).toInt();
      n_windows = input.substring(coma2 + 1).toInt();

      rstxt(WINDOW_FILE);
      rstxt(ENERGY_FILE);
      rstxt(ZCR_FILE);
      rstxt(TEST_FEATS_FILE);
      rstxt(COSTS_FILE);

      param = true;
      Serial.println("Parameters received correctly.");
      Serial.print("step: "); Serial.print(step);
      Serial.print(", win_len: "); Serial.print(win_len);
      Serial.print(", n_windows: "); Serial.println(n_windows);
      Serial1.println("1");
    } else {
      Serial.println("Error: Formato invalido.");
      Serial1.println("0");
    }
  }
}

//Función para procesar las entradas del vector señal
void procesarDatos() {
  // if (Serial1.available() > 0) {
      Serial1.println("3");  
      Serial.println("Esperando datos...");

      String input = Serial1.readStringUntil('\n');
      double sample = input.toDouble();
      Serial.println(input);

    if (input == "2") {
      Serial.println("Señal Concluida, Comenzando procesamiento...");
      gTest();
      aCSV();
      param = false;
      Serial.println("Fin del procesamiento.");
      return;
      }

    if (window_idx < step) {
      if (input == "" || sample == 0 && input != "0") {
        Serial.println("Error: dato no válido recibido.");
        return;
      }

      editxt(WINDOW_FILE, sample, window_idx);
      window_idx++;
    }

    if (window_idx == step) {
      Serial1.println("4");  
      Serial.println("Procesando ventana completa...");

      double c_energy = 0.0, c_zcr = 0.0;

      for (int i = 0; i < step; i++) {
        double windowSample = vertxt(WINDOW_FILE, i);
        c_energy += windowSample * windowSample;
      }

      for (int i = 1; i < step; i++) {
        double prevSample = vertxt(WINDOW_FILE, i - 1);
        double currentSample = vertxt(WINDOW_FILE, i);
        if ((prevSample > 0 && currentSample < 0) || (prevSample < 0 && currentSample > 0)) {
          c_zcr++;
        }
      }

      if (window_C < n_windows) {
        editxt(ENERGY_FILE, c_energy, window_C);
        editxt(ZCR_FILE, c_zcr, window_C);
        window_C++;
        Serial.print("Energía: ");
        Serial.println(c_energy);
        Serial.print("ZCR: ");
        Serial.println(c_zcr);
      }

      window_idx = 0;
      Serial1.println("3");  
      Serial.println("Listo para nuevos datos.");
    }
  // }
}

//FUnción para guardar los datos recopilados del test
void gTest() {
  Serial.println("Guardando Datos");
  for (int i = 0; i < window_C; i++) {
    double energy = vertxt(ENERGY_FILE, i);
    double zcr = vertxt(ZCR_FILE, i);
    
    // Store test features to file
    File Archivo = SD.open(TEST_FEATS_FILE, FILE_WRITE);
    if (Archivo) {
      Archivo.print(i);
      Archivo.print(",");
      Archivo.print(energy);
      Archivo.print(",");
      Archivo.println(zcr);
      Archivo.close();
    }
  }
}

//FUnción para realizar operaciones con la base de datos
void aCSV() {
  Serial.println("Analizando base de datos");
  File DB = SD.open("database.txt");
  if (!DB) {
    Serial.println("Error abriendo 'database.csv'.");
    return;
  }

  const int TOL_WIN = 5;
  double min_costoT = INFINITY;
  int Clase_det = -1;

  rstxt(COSTS_FILE);

  for (int Clase = 0; Clase < 10; Clase++) {
    DB.seek(0);
    Serial.println(Clase);

    while (DB.available()) {
    // Serial.println(DB.available());
      String linea = DB.readStringUntil('\n');
      linea.trim();


      int coma1 = linea.indexOf(',');
      int coma2 = linea.lastIndexOf(',');
      Serial.println(linea);
      Serial.println(coma1);
      Serial.println(coma2);
      if (coma1 == -1 || coma2 == -1) continue;

      if (linea.substring(coma1 + 1, coma2)=="Energy" || linea.substring(coma2 + 1)=="ZCR"||linea.substring(0, coma1)=="Class"){
        continue;
      } else{

      double csv_energy = linea.substring(coma1 + 1, coma2).toDouble();
      double csv_zcr = linea.substring(coma2 + 1).toDouble();
      int classLabel = linea.substring(0, coma1).toInt();

      if (classLabel == Clase) {//Serial.println("if1");
        for (int idx = 0; idx < window_C; idx++) {//Serial.println("for1");

          File Archivodtw = SD.open("dtw_temp.txt", FILE_WRITE);
          
          for (int i = 0; i <= step; i++) {//Serial.println("for2");
            for (int j = 0; j <= step; j++) {//Serial.println("for3");
              Archivodtw.print(i);
              Archivodtw.print(",");
              Archivodtw.print(j);
              Archivodtw.print(",");
              Archivodtw.println((i == 0 && j == 0) ? 0 : INFINITY);
            }
          }
          Archivodtw.close();

          double costof = DTWCalc(csv_energy, csv_zcr, idx);
          
          editxt(COSTS_FILE, costof, idx);
        }
      }
    }}
  }

  DB.close();

  double costoT = 0;
  for (int i = 0; i < window_C; i++) {Serial.println("for4");
    costoT += vertxt(COSTS_FILE, i);
  }

  if (costoT < min_costoT) {Serial.println("if2");
    min_costoT = costoT;
    Clase_det = result;
  }

  Serial.print("El numero es: ");
  Serial.println(Clase_det);
  Serial1.print("RESULTADO:"); 
  Serial1.println(Clase_det);
}

//FUnción para los calculos del DTW
double DTWCalc(double csv_energy, double csv_zcr, int idx) {
  Serial.println("Calculando DTW");
  double test_energy = vertxt(TEST_FEATS_FILE, idx);
  double test_zcr = vertxt(TEST_FEATS_FILE, idx + 1);

  const int TOL_WIN = 5;
  double min_cost = INFINITY;

  File Archivodtw = SD.open("dtw_temp.txt", FILE_READ);
  if (!Archivodtw) {
    Serial.println("No existe el Archivo DTW");
    return INFINITY;
  }

  while (Archivodtw.available()) {Serial.println("Abriendo matriz dtw");
    String linea = Archivodtw.readStringUntil('\n');
    int coma1 = linea.indexOf(',');
    int coma2 = linea.indexOf(',', coma1 + 1);
    Serial.println(linea);
    
    int i = linea.substring(0, coma1).toInt();
    int j = linea.substring(coma1 + 1, coma2).toInt();
    double val = linea.substring(coma2 + 1).toDouble();

    if (i > 0 && j > 0) {
      double cost = sqrt(pow(csv_energy - test_energy, 2) + pow(csv_zcr - test_zcr, 2));
      double min_mat = min(min(
        verDTWmat("dtw_temp.txt", i-1, j), 
        verDTWmat("dtw_temp.txt", i, j-1)
      ), verDTWmat("dtw_temp.txt", i-1, j-1));

      double n_val = cost + min_mat;
      edDTWmat("dtw_temp.txt", i, j, n_val);

      if (i == step && j == step) {
        min_cost = n_val;
      }
    }
  }

  Archivodtw.close();
  return min_cost;
}

//FUnción para la interacción con la "matriz DTW"
double verDTWmat(const char* txt, int i, int j) {
  File Archivodtw = SD.open(txt, FILE_READ);
  if (Archivodtw) {
    while (Archivodtw.available()) {
      String line = Archivodtw.readStringUntil('\n');
      int coma1 = line.indexOf(',');
      int coma2 = line.indexOf(',', coma1 + 1);
      
      int fileI = line.substring(0, coma1).toInt();
      int fileJ = line.substring(coma1 + 1, coma2).toInt();
      
      if (fileI == i && fileJ == j) {
        double value = line.substring(coma2 + 1).toDouble();
        Archivodtw.close();
        return value;
      }
    }
    Archivodtw.close();
  }
  return INFINITY;
}

//Función para la edición de la "MAtriz DTW"
void edDTWmat(const char* txt, int i, int j, double value) {
  File Archivodtw = SD.open(txt, FILE_WRITE);
  if (Archivodtw) {
    Archivodtw.print(i);
    Archivodtw.print(",");
    Archivodtw.print(j);
    Archivodtw.print(",");
    Archivodtw.println(value);
    Archivodtw.close();
  }
}

//Inicialización del programa
void setup() {
  Serial.begin(9600);
  Serial1.begin(115200);
  delay(1000);
  // Serial.println(SD.begin(10)); 
  if (!SD.begin(10)) {
    Serial.println("Error inicializando la tarjeta SD.");
    while (1);
  }
  Serial.println("Sistema iniciado.");
}

void loop() {
  
  if (!param) {
    recibirParametros();
  } else {
    delay(500);
    procesarDatos();
  }
}