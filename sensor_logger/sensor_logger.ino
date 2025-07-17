#include <DHT.h>

#define DHTPIN 4        // GPIO4
#define DHTTYPE DHT11   // DHT11
#define MQ135_PIN 1     // GPIO1

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  // Khởi tạo Serial với delay để ESP32-S3 ổn định
  delay(2000);
  Serial.begin(115200);
  while (!Serial) {
    delay(100);
  }
  
  Serial.println("=== ESP32-S3 Air Quality Monitor ===");
  Serial.println("Initializing sensors...");
  
  // Khởi tạo DHT11
  dht.begin();
  
  // Test DHT11
  delay(2000);
  float test_temp = dht.readTemperature();
  if (isnan(test_temp)) {
    Serial.println("DHT11 initialization failed!");
  } else {
    Serial.println("DHT11 initialized successfully");
  }
  
  // MQ135 warm-up
  Serial.println("MQ135 warming up (30s)...");
  for(int i = 30; i > 0; i--) {
    Serial.printf("Warm-up: %d seconds\n", i);
    delay(1000);
  }
  
  Serial.println("Starting data collection...");
  Serial.println("Format: temp,humidity,mq135_raw");
}

void loop() {
  // Đọc DHT11
  float temp = dht.readTemperature();
  float hum = dht.readHumidity();
  
  // Đọc MQ135
  int mq135_raw = analogRead(MQ135_PIN);
  
  // Kiểm tra dữ liệu hợp lệ
  if (isnan(temp) || isnan(hum)) {
    // KHÔNG in error lên Serial để tránh nhiễu CSV
    delay(2000);
    return;
  }
  
  // Kiểm tra giá trị hợp lý
  if (temp < -40 || temp > 80 || hum < 0 || hum > 100) {
    delay(2000);
    return;
  }
  
  // In PURE CSV data (chỉ 3 values, không có timestamp)
  Serial.print(temp, 1);
  Serial.print(",");
  Serial.print(hum, 1);
  Serial.print(",");
  Serial.println(mq135_raw);
  
  delay(1000);  // Thu thập mỗi giây
}

// Optional: Debug function (không gọi trong loop chính)
void debugSensors() {
  float temp = dht.readTemperature();
  float hum = dht.readHumidity();
  int mq135_raw = analogRead(MQ135_PIN);
  float mq135_voltage = mq135_raw * 3.3 / 4095.0;
  
  Serial.println("=== DEBUG INFO ===");
  Serial.printf("Temperature: %.1f°C\n", temp);
  Serial.printf("Humidity: %.1f%%\n", hum);
  Serial.printf("MQ135 Raw: %d\n", mq135_raw);
  Serial.printf("MQ135 Voltage: %.2fV\n", mq135_voltage);
  Serial.printf("Free Heap: %d bytes\n", ESP.getFreeHeap());
  Serial.println("==================");
}