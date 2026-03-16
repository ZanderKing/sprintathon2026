#include <SPI.h>
#include <SD.h>
#include <WiFi.h>
#include <WebServer.h>

// --- Wi-Fi Settings ---
const char* ssid = "Pavan";         // Change this!
const char* password = "Pavan1978"; // Change this!
WebServer server(80);

// --- Hardware Pins (Left Side Optimized) ---
const int ADC_PIN = 34;       // LM358 Op-Amp Output
const int BUTTON_PIN = 0;     // Built-in BOOT button

// Custom SPI Pins for Waveshare SD Module
const int SD_CLK = 26;
const int SD_MOSI = 27;
const int SD_MISO = 32;
const int SD_CS_PIN = 33; 

// --- System States ---
enum SystemState { READY_TO_RECORD, RECORDING, WIFI_MODE };
SystemState currentState = READY_TO_RECORD;

// --- Recording Settings ---
const int SAMPLE_RATE_HZ = 5000;                      
const int SAMPLE_INTERVAL_US = 1000000 / SAMPLE_RATE_HZ; // 200 microseconds
const int BUFFER_SIZE = 1000; // 0.2 seconds of audio per buffer

// --- Ping-Pong Buffers ---
uint16_t bufferA[BUFFER_SIZE];
uint16_t bufferB[BUFFER_SIZE];

volatile bool useBufferA = true;      
volatile bool readyToWriteA = false;  
volatile bool readyToWriteB = false;

unsigned long totalSamplesRecorded = 0;
File dataFile;
TaskHandle_t SDWriteTask;

// Button debounce tracking
unsigned long lastButtonPress = 0;

void setup() {
  Serial.begin(115200);
  pinMode(ADC_PIN, INPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  // 1. Keep Wi-Fi OFF initially to prevent electrical noise in the ADC
  WiFi.mode(WIFI_OFF);

  // 2. Initialize the SD Card on the custom Left-Side Pins
  Serial.print("Initializing SD card on custom SPI pins...");
  SPI.begin(SD_CLK, SD_MISO, SD_MOSI, SD_CS_PIN);
  
  if (!SD.begin(SD_CS_PIN)) {
    Serial.println("\nCard Mount Failed! Check your wiring.");
    while (1); // Halt system if no SD card
  }
  Serial.println(" SD Card mounted.");

  // 3. Launch the SD writing task onto Core 0
  xTaskCreatePinnedToCore(sdWriteLoop, "SD_Task", 10000, NULL, 1, &SDWriteTask, 0);

  Serial.println("\n=========================================");
  Serial.println("SYSTEM READY.");
  Serial.println("Press the BOOT button to START recording.");
  Serial.println("=========================================");
}

void loop() {
  // --- Button State Machine ---
  if (digitalRead(BUTTON_PIN) == LOW && millis() - lastButtonPress > 500) {
    lastButtonPress = millis();
    
    if (currentState == READY_TO_RECORD) {
      startRecording();
    } 
    else if (currentState == RECORDING) {
      stopRecordingAndStartWiFi();
    }
    // If in WIFI_MODE, do nothing. Reset board to record again.
  }

  // --- Run tasks based on the current state ---
  if (currentState == RECORDING) {
    sampleAudio(); // Core 1: Aggressively sample the op-amp
  } 
  else if (currentState == WIFI_MODE) {
    server.handleClient(); // Core 1: Listen for browser downloads
  }
}

// ==========================================
// Phase 1: RECORDING FUNCTIONS (Core 1)
// ==========================================

void startRecording() {
  // Overwrite old file to keep data clean
  if (SD.exists("/crepitus.csv")) {
    SD.remove("/crepitus.csv");
  }

  dataFile = SD.open("/crepitus.csv", FILE_WRITE);
  if (dataFile) {
    dataFile.println("Timestamp,Signal");
    useBufferA = true;
    readyToWriteA = false;
    readyToWriteB = false;
    totalSamplesRecorded = 0;
    currentState = RECORDING;
    Serial.println("\n[RECORDING STARTED] - Tracking to SD Card. Press BOOT to stop.");
  } else {
    Serial.println("Failed to open SD file for writing!");
  }
}

void sampleAudio() {
  static unsigned long nextSampleTime = micros();
  static int bufferIndex = 0;

  // Busy wait for absolute microsecond precision
  while (micros() < nextSampleTime) {}
  nextSampleTime += SAMPLE_INTERVAL_US;

  // Read ADC and store in active RAM buffer
  if (useBufferA) {
    bufferA[bufferIndex] = analogRead(ADC_PIN);
  } else {
    bufferB[bufferIndex] = analogRead(ADC_PIN);
  }
  
  bufferIndex++;

  // Swap buffers when full and trigger Core 0 to save
  if (bufferIndex >= BUFFER_SIZE) {
    bufferIndex = 0;
    if (useBufferA) {
      useBufferA = false;      
      readyToWriteA = true;    
    } else {
      useBufferA = true;       
      readyToWriteB = true;    
    }
  }
}

// ==========================================
// SD Card Background Task (Core 0)
// ==========================================

void sdWriteLoop(void * parameter) {
  for (;;) {
    if (readyToWriteA) {
      writeBufferToSD(bufferA);
      readyToWriteA = false; 
    }
    if (readyToWriteB) {
      writeBufferToSD(bufferB);
      readyToWriteB = false;
    }
    vTaskDelay(pdMS_TO_TICKS(1)); // Prevents Core 0 watchdog crash
  }
}

void writeBufferToSD(uint16_t* buffer) {
  if (!dataFile) return; 
  
  float time_increment = 1000.0 / (float)SAMPLE_RATE_HZ;
  
  for (int i = 0; i < BUFFER_SIZE; i++) {
    float current_time_ms = totalSamplesRecorded * time_increment;
    dataFile.print(current_time_ms, 2); // Prints timestamp with 2 decimals
    dataFile.print(",");
    dataFile.println(buffer[i]);
    totalSamplesRecorded++;
  }
  
  // Physically push data to the SD card every 5 buffers
  if (totalSamplesRecorded % (BUFFER_SIZE * 5) == 0) {
    dataFile.flush();
  }
}

// ==========================================
// Phase 2: WI-FI & WEB SERVER FUNCTIONS
// ==========================================

void stopRecordingAndStartWiFi() {
  currentState = WIFI_MODE;
  delay(100); // Wait for the final RAM buffer to finish dumping
  
  if (dataFile) {
    dataFile.close();
    Serial.println("\n[RECORDING STOPPED] - File saved safely to SD Card.");
  }

  // Boot up the Wi-Fi Radio
  Serial.print("Booting up Wi-Fi Module");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\n[WI-FI CONNECTED!]");
  Serial.print("Go to this URL in your browser: http://");
  Serial.println(WiFi.localIP());

  // Setup Server Routes
  server.on("/", HTTP_GET, handleRoot);
  server.on("/download", HTTP_GET, handleDownload);
  server.begin();
}

void handleRoot() {
  String html = "<html><body style='font-family: Arial; text-align: center; margin-top: 50px;'>";
  html += "<h2>Osteoarthritis Data Portal</h2>";
  html += "<p>Total Samples Recorded: " + String(totalSamplesRecorded) + "</p>";
  float secondsRecorded = (float)totalSamplesRecorded / SAMPLE_RATE_HZ;
  html += "<p>Total Audio Length: " + String(secondsRecorded, 2) + " seconds</p>";
  html += "<a href=\"/download\"><button style=\"padding:15px 30px; font-size:18px; background-color:#4CAF50; color:white; border:none; border-radius:5px;\">Download crepitus.csv</button></a>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void handleDownload() {
  if (!SD.exists("/crepitus.csv")) {
    server.send(404, "text/plain", "File not found on SD Card!");
    return;
  }
  
  File file = SD.open("/crepitus.csv", FILE_READ);
  server.streamFile(file, "text/csv");
  file.close();
}