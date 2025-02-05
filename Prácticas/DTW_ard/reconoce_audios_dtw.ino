

#include <WiFi.h>
#include <math.h>

const char* ssid = "INFINITUMAE55_2.4";
const char* password = "zNAutBwZxc";
WiFiServer server(80);

const int TAMANO_MAXIMO = 100;
int ventana[TAMANO_MAXIMO];
int tamanoVentana = 0;

long cero [] = {4344, 67922, 59313, 73806, 73316, 54088, 74951, 64527, 56181, 23291, 74515, 37210, 19488,
                99, 67, 26, 35, 46, 45, 50, 52, 40, 42, 64, 56, 38, 31, 1};
long uno[] = {3730, 25700, 35122, 47865, 49540, 65902, 38092, 11737, 11350, 16169, 17078, 30951, 22700, 11251, 
              16, 24, 28, 24, 28, 39, 28, 42, 43, 38, 34, 22, 39, 33};
long dos[] = {2758, 45718, 66046, 79971, 79239, 60836, 46576, 32462, 17473,
              26, 35, 27, 29, 31, 37, 34, 38, 24};
long tres [] = {49241, 16310, 68302, 99194, 88043, 74715, 54838, 31605, 4349, 
                56, 68, 50, 34, 44, 40, 57, 36, 27};
long cuatro [] = {5495, 7259, 8447, 16150, 46330, 39874, 52665, 54883, 70866, 93960, 40987, 
                  20, 19, 22, 24, 28, 39, 54, 56, 62, 58, 46};
long cinco[] = {16907, 29198, 16681, 11951, 9144, 6862, 4493, 5931, 5354, 21868, 61899, 55114, 20581, 3011, 
                52, 32, 29, 24, 32, 39, 36, 33, 36, 28, 33, 25, 24, 13};
long seis [] = {3913, 43489, 97634, 79415, 75472, 52735, 19567, 11248, 13614, 4872, 4164, 716, 
                63, 23, 22, 30, 40, 27, 31, 26, 21, 21, 14, 4};
long siete[] = {14035, 16227, 23024, 60784, 98748, 83063, 88907, 52567, 41337, 32867, 12016, 2483, 
                28, 22, 22, 23, 34, 42, 54, 52, 49, 36, 32, 9};
long ocho [] = {27844, 77360, 71286, 87850, 55275, 42250, 32311, 24137, 22455, 39200, 26991, 14018, 11300, 
                32, 34, 39, 37, 33, 26, 37, 26, 35, 41, 34, 31, 31};
long nueve[] = {2629, 4201, 4036, 3245, 3099, 5466, 7503, 9725, 15334, 23369, 26862, 22920, 23755, 20395, 21794, 18889, 24259, 44321, 30521, 60498, 11277, 5759, 
                30, 27, 32, 28, 29, 31, 30, 29, 33, 35, 35, 37, 53, 40, 43, 64, 58, 66, 58, 40, 30, 20};


long concatenado_todos[200];
bool c = true;
long energias[100];  
int zcrs[100];        
int contadorVentanas = 0;

void convertirStringAArray(String str, int* array, int& tamaño) {
    int i = 0, startIndex = 0;
    for (int j = 0; j < str.length(); j++) {
        if (str.charAt(j) == ' ' || j == str.length() - 1) {
            array[i] = str.substring(startIndex, j).toInt();
            i++;
            startIndex = j + 1;
        }
    }
    tamaño = i;
}

long calcularEnergia(int* ventana, int tamaño) {
    long energia = 0;
    for (int i = 0; i < tamaño; i++) {
        energia += ventana[i] * ventana[i];
    }
    return energia;
}

int calcularZCR(int* ventana, int tamaño) {
    int zcr = 0;
    for (int i = 1; i < tamaño; i++) {
        if ((ventana[i-1] >= 0 && ventana[i] < 0) || (ventana[i-1] < 0 && ventana[i] >= 0)) {
            zcr++;
        }
    }
    return zcr;
}

void concatenarEnergiaZCR() {
    for (int i = 0; i < contadorVentanas; i++) {
        concatenado_todos[i] = energias[i];
    }
    for (int i = 0; i < contadorVentanas; i++) {
        concatenado_todos[contadorVentanas + i] = zcrs[i];
    }
}

String compararDTW() {
    long* referencias[] = {cero, uno, dos, tres, cuatro, cinco, seis, siete, ocho, nueve};
    int longitudes[] = {
        sizeof(cero)/sizeof(cero[0]), 
        sizeof(uno)/sizeof(uno[0]),
        sizeof(dos)/sizeof(dos[0]),
        sizeof(tres)/sizeof(tres[0]),
        sizeof(cuatro)/sizeof(cuatro[0]),
        sizeof(cinco)/sizeof(cinco[0]),
        sizeof(seis)/sizeof(seis[0]),
        sizeof(siete)/sizeof(siete[0]),
        sizeof(ocho)/sizeof(ocho[0]),
        sizeof(nueve)/sizeof(nueve[0])
    };

    float mejorDistancia = INFINITY;
    int mejorIndice = -1;
    int totalDatos = contadorVentanas * 2;

    for (int k = 0; k < 10; k++) {
        float dtw[2][200] = {0};
        int ref_len = longitudes[k];

        // Inicializar primera fila
        for (int j = 0; j < totalDatos; j++) {
            dtw[0][j] = abs(concatenado_todos[j] - referencias[k][0]);
        }

        // Calcular matriz DTW
        for (int i = 1; i < ref_len; i++) {
            for (int j = 0; j < totalDatos; j++) {
                float costo = abs(concatenado_todos[j] - referencias[k][i]);
                dtw[i % 2][j] = costo + min(
                    dtw[(i-1) % 2][j],
                    min(
                        j > 0 ? dtw[i % 2][j-1] : INFINITY,
                        j > 0 ? dtw[(i-1) % 2][j-1] : INFINITY
                    )
                );
            }
        }

        // Encontrar la distancia mínima
        float distancia = dtw[(ref_len-1) % 2][totalDatos-1];
        if (distancia < mejorDistancia) {
            mejorDistancia = distancia;
            mejorIndice = k;
        }
    }

    return "La señal se parece mas a la variable " + String(mejorIndice);
}

void setup() {
    Serial.begin(115200);
    
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    
    Serial.println("\nWiFi conectado");
    Serial.println("Dirección IP: ");
    Serial.println(WiFi.localIP());
    
    server.begin();
}

void loop() {
    WiFiClient client = server.available();
    
    if (client) {
        while (client.connected()) {
            if (client.available()) {
                String datos = client.readStringUntil('\n');
                
                if (datos.startsWith("F")) {
                    concatenarEnergiaZCR();
                    String resultado = compararDTW();
                    Serial.println(resultado);
                    client.println(resultado);
                    c = false;
                    contadorVentanas = 0;
                }

                if (c == true){
                    convertirStringAArray(datos, ventana, tamanoVentana);
                    long energia = calcularEnergia(ventana, tamanoVentana);
                    int zcr = calcularZCR(ventana, tamanoVentana);

                    energias[contadorVentanas] = energia;
                    zcrs[contadorVentanas] = zcr;
                    contadorVentanas++;
                }

                client.println(String(contadorVentanas) + " ventanas recibidas correctamente");
                Serial.print(contadorVentanas);
                Serial.println(" ventanas recibidas correctamente");
                c = true;
            }
        }
        client.stop();
    }
}




/*
#include <WiFi.h>

const char* ssid = "INFINITUMAE55_2.4";
const char* password = "zNAutBwZxc";
WiFiServer server(80);

const int TAMANO_MAXIMO = 100; // Tamaño máximo de la ventana
int ventana[TAMANO_MAXIMO]; // Arreglo para almacenar los valores de la ventana
int tamanoVentana = 0; // Variable para almacenar el tamaño real de la ventana

long cero [] = {4344, 67922, 59313, 73806, 73316, 54088, 74951, 64527, 56181, 23291, 74515, 37210, 19488,
                99, 67, 26, 35, 46, 45, 50, 52, 40, 42, 64, 56, 38, 31, 1};
long uno[] = {3730, 25700, 35122, 47865, 49540, 65902, 38092, 11737, 11350, 16169, 17078, 30951, 22700, 11251, 
              16, 24, 28, 24, 28, 39, 28, 42, 43, 38, 34, 22, 39, 33};
long dos[] = {2758, 45718, 66046, 79971, 79239, 60836, 46576, 32462, 17473,
              26, 35, 27, 29, 31, 37, 34, 38, 24};
long tres [] = {49241, 16310, 68302, 99194, 88043, 74715, 54838, 31605, 4349, 
                56, 68, 50, 34, 44, 40, 57, 36, 27};
long cuatro [] = {5495, 7259, 8447, 16150, 46330, 39874, 52665, 54883, 70866, 93960, 40987, 
                  20, 19, 22, 24, 28, 39, 54, 56, 62, 58, 46};
long cinco[] = {16907, 29198, 16681, 11951, 9144, 6862, 4493, 5931, 5354, 21868, 61899, 55114, 20581, 3011, 
                52, 32, 29, 24, 32, 39, 36, 33, 36, 28, 33, 25, 24, 13};
long seis [] = {3913, 43489, 97634, 79415, 75472, 52735, 19567, 11248, 13614, 4872, 4164, 716, 
                63, 23, 22, 30, 40, 27, 31, 26, 21, 21, 14, 4};
long siete[] = {14035, 16227, 23024, 60784, 98748, 83063, 88907, 52567, 41337, 32867, 12016, 2483, 
                28, 22, 22, 23, 34, 42, 54, 52, 49, 36, 32, 9};
long ocho [] = {27844, 77360, 71286, 87850, 55275, 42250, 32311, 24137, 22455, 39200, 26991, 14018, 11300, 
                32, 34, 39, 37, 33, 26, 37, 26, 35, 41, 34, 31, 31};
long nueve[] = {2629, 4201, 4036, 3245, 3099, 5466, 7503, 9725, 15334, 23369, 26862, 22920, 23755, 20395, 21794, 18889, 24259, 44321, 30521, 60498, 11277, 5759, 
                30, 27, 32, 28, 29, 31, 30, 29, 33, 35, 35, 37, 53, 40, 43, 64, 58, 66, 58, 40, 30, 20};



long concatenado_todos[200]; // Arreglo para almacenar las energías y ZCR concatenados
bool c = true;
// Arreglos para almacenar la energía y el ZCR de cada ventana (sin tamaño fijo)
long energias[100];  
int zcrs[100];        
int contadorVentanas = 0;  // Contador de ventanas procesadas

//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void convertirStringAArray(String str, int* array, int& tamaño) {
    // Partimos el string por espacios
    int i = 0;
    int startIndex = 0;
    for (int j = 0; j < str.length(); j++) {
        if (str.charAt(j) == ' ' || j == str.length() - 1) {
            String numStr = str.substring(startIndex, j);
            array[i] = numStr.toInt(); // Convertimos la parte del string a entero
            i++;
            startIndex = j + 1;
        }
    }
    tamaño = i; // Devolvemos el tamaño del arreglo
}

// Función para calcular la energía
long calcularEnergia(int* ventana, int tamaño) {
    long energia = 0;
    for (int i = 0; i < tamaño; i++) {
        energia += ventana[i] * ventana[i]; // Sumamos el cuadrado de cada muestra
    }
    return energia;
}

// Función para calcular el cruce por cero (ZCR)
int calcularZCR(int* ventana, int tamaño) {
    int zcr = 0;
    for (int i = 1; i < tamaño; i++) {
        // Comprobamos si el signo cambia entre dos muestras consecutivas
        if ((ventana[i-1] >= 0 && ventana[i] < 0) || (ventana[i-1] < 0 && ventana[i] >= 0)) {
            zcr++; // Si cambia de signo, contamos un cruce por cero
        }
    }
    return zcr;
}

// Función para concatenar las energías y ZCR en un solo arreglo
void concatenarEnergiaZCR() {
    for (int i = 0; i < contadorVentanas; i++) {
        concatenado_todos[i] = energias[i]; // Copiar energías
    }
    for (int i = 0; i < contadorVentanas; i++) {
        concatenado_todos[contadorVentanas + i] = zcrs[i]; // Copiar ZCRs
    }
}

//=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=-=--=-

void setup() {
  Serial.begin(115200);
  
  // Conectar al WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.println("WiFi conectado");
  Serial.println("Dirección IP: ");
  Serial.println(WiFi.localIP());
  
  server.begin();
}

void loop() {
  WiFiClient client = server.available();
  
  if (client) {
    while (client.connected()) {
      if (client.available()) {
        String datos = client.readStringUntil('\n');
        
        // Comprobamos si los datos contienen la letra "F"
        if (datos.startsWith("F")) {
            concatenarEnergiaZCR(); // Concatenamos energía y ZCR
            client.println("Datos completos");
            // Mostrar los cruces por cero almacenados
            Serial.println("Datos almacenados:");
            for (int i = 0; i < contadorVentanas*2; i++) {
              Serial.print(concatenado_todos[i]);
              Serial.print(", ");
            }
            Serial.println();
            c = false;
            contadorVentanas = 0;
        }

        if (c == true){
            // Convertimos el string de datos a un array de enteros
            convertirStringAArray(datos, ventana, tamanoVentana);

            // Calculamos la energía y el ZCR de la ventana
            long energia = calcularEnergia(ventana, tamanoVentana);
            int zcr = calcularZCR(ventana, tamanoVentana);

            // Almacenamos la energía y ZCR en los arreglos
            energias[contadorVentanas] = energia;
            zcrs[contadorVentanas] = zcr;
            contadorVentanas++;
        }

        //   Imprimimos los resultados
        // Serial.print("Energía: ");
        // Serial.println(energia);
        // Serial.print("Cruces por cero: ");
        // Serial.println(zcr);

        // Enviar respuesta al cliente
        client.println(String(contadorVentanas) + " ventanas recibidas correctamente");
        Serial.print(contadorVentanas);
        Serial.println(" ventanas recibidas correctamente");
        c = true;
      }
    }
    client.stop();

    // // Mostrar las energías almacenadas
    // Serial.println("Energías almacenadas:");
    // for (int i = 0; i < contadorVentanas; i++) {
    //   Serial.print(energias[i]);
    //   Serial.print(" ");
    // }
    // Serial.println();

    // // Mostrar los cruces por cero almacenados
    // Serial.println("Cruces por cero almacenados:");
    // for (int i = 0; i < contadorVentanas; i++) {
    //   Serial.print(zcrs[i]);
    //   Serial.print(" ");
    // }
    // Serial.println();
  }
}
*/




// #include <WiFi.h>

// const char* ssid = "INFINITUMAE55_2.4";
// const char* password = "zNAutBwZxc";
// WiFiServer server(80);

// const int TAMANO_MAXIMO = 100;
// int ventana[TAMANO_MAXIMO];
// int tamanoVentana = 0;

// float energias[100];  
// int zcrs[100];        
// int contadorVentanas = 0;  

// // Variables para almacenar datos concatenados de cada índice
// float datos_cero[200];
// float datos_uno[200];
// float datos_dos[200];
// float datos_tres[200];
// float datos_cuatro[200];
// float datos_cinco[200];
// float datos_seis[200];
// float datos_siete[200];
// float datos_ocho[200];
// float datos_nueve[200];

// int longitud_cero = 0;
// int longitud_uno = 0;
// int longitud_dos = 0;
// int longitud_tres = 0;
// int longitud_cuatro = 0;
// int longitud_cinco = 0;
// int longitud_seis = 0;
// int longitud_siete = 0;
// int longitud_ocho = 0;
// int longitud_nueve = 0;

// void convertirStringAArray(String str, int* array, int& tamaño) {
//     int i = 0;
//     int startIndex = 0;
//     for (int j = 0; j < str.length(); j++) {
//         if (str.charAt(j) == ' ' || j == str.length() - 1) {
//             String numStr = str.substring(startIndex, j);
//             array[i] = numStr.toInt();
//             i++;
//             startIndex = j + 1;
//         }
//     }
//     tamaño = i;
// }

// float calcularEnergia(int* ventana, int tamaño) {
//     float energia = 0;
//     for (int i = 0; i < tamaño; i++) {
//         energia += ventana[i] * ventana[i];
//     }
//     return energia;
// }

// int calcularZCR(int* ventana, int tamaño) {
//     int zcr = 0;
//     for (int i = 1; i < tamaño; i++) {
//         if ((ventana[i-1] >= 0 && ventana[i] < 0) || (ventana[i-1] < 0 && ventana[i] >= 0)) {
//             zcr++;
//         }
//     }
//     return zcr;
// }

// void guardarDatosConcatenados(int indice) {
//     float* arreglo_destino;
//     int* longitud_destino;

//     switch(indice) {
//         case 0: arreglo_destino = datos_cero; longitud_destino = &longitud_cero; break;
//         case 1: arreglo_destino = datos_uno; longitud_destino = &longitud_uno; break;
//         case 2: arreglo_destino = datos_dos; longitud_destino = &longitud_dos; break;
//         case 3: arreglo_destino = datos_tres; longitud_destino = &longitud_tres; break;
//         case 4: arreglo_destino = datos_cuatro; longitud_destino = &longitud_cuatro; break;
//         case 5: arreglo_destino = datos_cinco; longitud_destino = &longitud_cinco; break;
//         case 6: arreglo_destino = datos_seis; longitud_destino = &longitud_seis; break;
//         case 7: arreglo_destino = datos_siete; longitud_destino = &longitud_siete; break;
//         case 8: arreglo_destino = datos_ocho; longitud_destino = &longitud_ocho; break;
//         case 9: arreglo_destino = datos_nueve; longitud_destino = &longitud_nueve; break;
//     }

//     // Copiar ZCRs
//     for (int i = 0; i < contadorVentanas; i++) {
//         arreglo_destino[*longitud_destino] = zcrs[i];
//         (*longitud_destino)++;
//     }

//     // Copiar Energías
//     for (int i = 0; i < contadorVentanas; i++) {
//         arreglo_destino[*longitud_destino] = energias[i];
//         (*longitud_destino)++;
//     }
// }

// void setup() {
//     Serial.begin(115200);
    
//     WiFi.begin(ssid, password);
//     while (WiFi.status() != WL_CONNECTED) {
//         delay(500);
//         Serial.println("Conectando---------")
//         Serial.print(".");
//     }
    
//     Serial.println("");
//     Serial.println("WiFi conectado");
//     Serial.println("Dirección IP: ");
//     Serial.println(WiFi.localIP());
    
//     server.begin();
// }

// void loop() {
//     WiFiClient client = server.available();
    
//     if (client) {
//         while (client.connected()) {
//             if (client.available()) {
//                 String datos = client.readStringUntil('\n');
//                 convertirStringAArray(datos, ventana, tamanoVentana);

//                 // Si es un flyer de 10 datos
//                 if (tamanoVentana == 10) {
//                     // Encontrar la posición del 1
//                     int indice = -1;
//                     for (int i = 0; i < 10; i++) {
//                         if (ventana[i] == 1) {
//                             indice = i;
//                             break;
//                         }
//                     }

//                     // Si es un flyer válido
//                     if (indice != -1) {
//                         // Guardar datos concatenados en la variable correspondiente
//                         guardarDatosConcatenados(indice);
                        
//                         // Reiniciar contador de ventanas
//                         contadorVentanas = 0;
//                     }
//                 }
//                 // Si es ventana de 100 datos
//                 else if (tamanoVentana == 100) {
//                     float energia = calcularEnergia(ventana, tamanoVentana);
//                     int zcr = calcularZCR(ventana, tamanoVentana);

//                     energias[contadorVentanas] = energia;
//                     zcrs[contadorVentanas] = zcr;
//                     contadorVentanas++;

//                     client.println("Datos recibidos correctamente");
//                 }
//             }
//         }
//         client.stop();
//     }
// }


