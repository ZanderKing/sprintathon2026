const int ADC_PIN = 34;       // LM358 Op-Amp Output
const int BUTTON_PIN = 0;     // Built-in BOOT button

// --- Recording Settings ---
const int SAMPLE_RATE_HZ = 5000;                      
const int SAMPLE_INTERVAL_US = 1000000 / SAMPLE_RATE_HZ; // 200 microseconds

bool isRecording = false;
unsigned long lastButtonPress = 0;
unsigned long totalSamplesRecorded = 0;

void setup() {
  // Use a high baud rate to support 5000Hz sampling without bottlenecking
  // 5000 samples/sec * ~15 chars/sample = 75,000 bytes/sec. 921600 baud handles ~92,000 bytes/sec.
  Serial.begin(921600); 
  pinMode(ADC_PIN, INPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  Serial.println("\n=========================================");
  Serial.println("SYSTEM READY.");
  Serial.println("Press the BOOT button to START/STOP recording.");
  Serial.println("=========================================");
}

void loop() {
  // --- Button State Machine ---
  if (digitalRead(BUTTON_PIN) == LOW && millis() - lastButtonPress > 500) {
    lastButtonPress = millis();
    isRecording = !isRecording;
    
    if (isRecording) {
      Serial.println("START");
      totalSamplesRecorded = 0;
    } else {
      Serial.println("STOP");
    }
  }

  // --- Sample & Send Data ---
  if (isRecording) {
    static unsigned long nextSampleTime = micros();
    
    // Busy wait for absolute microsecond precision
    while (micros() < nextSampleTime) {}
    nextSampleTime += SAMPLE_INTERVAL_US;

    int adcValue = analogRead(ADC_PIN);
    float time_increment = 1000.0 / (float)SAMPLE_RATE_HZ;
    float current_time_ms = totalSamplesRecorded * time_increment;
    
    // Print in CSV format directly to Serial
    Serial.print(current_time_ms, 2);
    Serial.print(",");
    Serial.println(adcValue);
    
    totalSamplesRecorded++;
  }
}
