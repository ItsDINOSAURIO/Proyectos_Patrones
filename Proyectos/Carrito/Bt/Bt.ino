#include <SoftwareSerial.h>

SoftwareSerial bluetooth(8, 9); // RX 10 TX 11


void setup() {
  bluetooth.begin(9600); 
  Serial.begin(9600);    
  pinMode(2,OUTPUT); //IN1
  pinMode(3,OUTPUT); //IN2
  pinMode(4,OUTPUT); //IN3
  pinMode(5,OUTPUT); //IN4
  pinMode(10,OUTPUT); //ENA
  pinMode(11,OUTPUT); //ENB
  analogWrite(10,125);
  analogWrite(11,150);

  Serial.println("Arduino listo para recibir datos vía Bluetooth.");
  bluetooth.println("Módulo Bluetooth conectado a Arduino.");
}

void loop() {
  if (bluetooth.available()) {
    String msg = bluetooth.readStringUntil('\n'); // Leer mensaje hasta nueva línea
    int msgL=msg.length();
    if (msgL>2){
      msg=msg.substring(0,2);
    }
    Serial.println("Mensaje recibido: " + msg); 

    String resp = "Arduino recibió: " + msg;
    bluetooth.println(resp);

    if (msg == "00") {//Delante
      reCanales();
      digitalWrite(2, HIGH); 
      digitalWrite(5, HIGH); 
      delay(600);
      reCanales();
      bluetooth.println("Delante.");
    } else if (msg == "10") {//Izquierda
      reCanales();
      digitalWrite(2, HIGH);
      delay(200);
      reCanales();
      bluetooth.println("Giro Izquierda.");
    } else if (msg == "01"){//Derecha
      reCanales();
      digitalWrite(5, HIGH);
      delay(200);
      reCanales();
      bluetooth.println("Giro Derecha.");
    } else if (msg == "11"){//Atrás
      reCanales();
      digitalWrite(3, HIGH); 
      digitalWrite(4, HIGH); 
      delay(600);
      reCanales();
      bluetooth.println("Atrás.");
    } else {bluetooth.println("Comando no reconocido.");}
  }

}

void reCanales(){
      digitalWrite(2, LOW);
      digitalWrite(3, LOW);
      digitalWrite(4, LOW);
      digitalWrite(5, LOW);
}