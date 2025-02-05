// void setup() {
// Serial.begin(9600);
// }

// void loop() {
// Serial.println("Hola mundo");
// delay(1000);
// }

// void setup() {
//     Serial.begin(9600);
// }

// void loop() {
//     if (Serial.available() > 0) {
//         // Leer el mensaje de Python
//         String mensaje = Serial.readString();
//         Serial.println("Mensaje recibido: " + mensaje);
        
//         // Aquí puedes realizar acciones basadas en el mensaje recibido
//         if (mensaje == "Mensaje desde Python") {
//             Serial.println("¡Mensaje específico recibido!");
//             // Realiza alguna acción en el Arduino
//         }
//     }
// }


// void setup() {
//     Serial.begin(9600);
// }

// void loop() {
//     if (Serial.available() > 0) {
//         String mensaje = Serial.readString();
//         Serial.println("Mensaje recibido: " + mensaje);
//     }
//     // Enviar un mensaje cada segundo para confirmar que Arduino envía datos
//     Serial.println("Arduino funcionando");
//     delay(1000);
// }

void setup() {
    Serial.begin(9600);
}

void loop() {
    if (Serial.available() > 0) {
        // Leer el mensaje de Python
        String mensaje = Serial.readString();
        
        // Enviar una respuesta de confirmación
        Serial.println("Mensaje recibido en Arduino: " + mensaje);
        
        // Puedes agregar más lógica aquí si quieres realizar acciones específicas
    }
}
