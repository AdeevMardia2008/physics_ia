#include <Servo.h>  

Servo myServo;
int servoPin = 14;

void setup() {
    myServo.attach(servoPin);
    myServo.write(0);
    delay(1000);
    myServo.write(180);
    delay(1000);
    myServo.detach();
}

void loop() {
    
}
