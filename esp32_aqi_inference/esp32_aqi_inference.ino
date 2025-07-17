/*
ESP32 TensorFlow Lite with Embedded Files
No SD card or SPIFFS needed!
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "config.h"  // WiFi and Telegram credentials

// TensorFlow Lite
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Generated files (create these with Python script above)
#include "esp32_aqi_model.h"    // Model parameters
#include "embedded_data.h"      // Embedded model + CSV data

// === TENSORFLOW LITE SETUP ===
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// === DATA STRUCTURES ===
struct TestSample {
    int sample_id;
    String scenario;
    float temp_sensor;
    float humidity_sensor;
    float mq135_raw;
    float mq135_normalized;
    float comfort_index;
    float temp_diff;
    float humidity_diff;
    float pm_ratio;
};

std::vector<TestSample> test_samples;
int current_sample = 0;

void setup() {
    Serial.begin(115200);
    delay(2000);
    
    Serial.println("🚀 ESP32 with Embedded AI Model");
    Serial.println("No SD card needed!");
    for (int i = 0; i < 40; i++) Serial.print("=");
    Serial.println();
 
    // === STEP 1: Connect WiFi ===
    Serial.println("\n📡 Connecting to WiFi...");
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.print(".");
    }
    Serial.println("\n✅ WiFi connected");
    
    // === STEP 2: Load Embedded Model ===
    Serial.println("\n🤖 Loading embedded TensorFlow Lite model...");
    if (!loadEmbeddedTFLiteModel()) {
        Serial.println("❌ Model loading failed!");
        return;
    }
    Serial.println("✅ Model loaded successfully");
    
    // === STEP 3: Parse Embedded CSV ===
    Serial.println("\n📋 Parsing embedded test data...");
    if (!parseEmbeddedCSV()) {
        Serial.println("❌ CSV parsing failed!");
        return;
    }
    Serial.printf("✅ Parsed %d test samples\n", test_samples.size());
    
    // === STEP 4: Send startup message ===
    sendTelegramMessage("🚀 *ESP32 AI Started!*\n\n🤖 Embedded TensorFlow Lite model\n📊 " + 
                       String(test_samples.size()) + " test samples loaded\n⚡ No external storage needed!");
    
    Serial.println("\n🎯 Starting predictions...");
}

void loop() {
    if (current_sample < test_samples.size()) {
        runPrediction(current_sample);
        current_sample++;
        delay(10000);  // 10 seconds between predictions
    } else {
        Serial.println("\n🎉 All predictions completed!");
        sendTelegramMessage("🎉 *All " + String(test_samples.size()) + " predictions completed!*\n\n🤖 ESP32 AI system working perfectly");
        current_sample = 0;  // Reset
        delay(60000);  // Wait 1 minute before restart
    }
}

bool loadEmbeddedTFLiteModel() {
    // Use embedded model data (no file system needed!)
    model = tflite::GetModel(embedded_model_data);
    
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("❌ Model version mismatch: %d vs %d\n", 
                     model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    
    // Setup resolver with required ops
    static tflite::MicroMutableOpResolver<5> resolver;  // Increase to 5 ops
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddDequantize();
    resolver.AddQuantize();      // Add if needed
    resolver.AddReshape();       // Add if needed
    
    // Create interpreter with new API (no error_reporter parameter)
    interpreter = new tflite::MicroInterpreter(
        model, 
        resolver, 
        tensor_arena, 
        kTensorArenaSize
        // Remove error_reporter parameter
    );
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("❌ Tensor allocation failed");
        return false;
    }
    
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.printf("📊 Model ready: Input[%d,%d], Output[%d,%d]\n",
                  input->dims->data[0], input->dims->data[1],
                  output->dims->data[0], output->dims->data[1]);
    
    return true;
}

bool parseEmbeddedCSV() {
    // Get embedded CSV data (no file system needed!)
    String csv_content = getEmbeddedCSV();
    
    // Split into lines
    int line_start = 0;
    int line_end = 0;
    bool first_line = true;  // Skip header
    
    while ((line_end = csv_content.indexOf('\n', line_start)) != -1) {
        String line = csv_content.substring(line_start, line_end);
        line.trim();
        
        if (!first_line && line.length() > 0) {
            TestSample sample = parseCSVLine(line);
            if (sample.sample_id > 0) {
                test_samples.push_back(sample);
            }
        }
        
        first_line = false;
        line_start = line_end + 1;
    }
    
    return test_samples.size() > 0;
}

TestSample parseCSVLine(String line) {
    TestSample sample;
    sample.sample_id = 0;  // Default invalid
    
    // Parse CSV fields
    int field = 0;
    int start = 0;
    int comma_pos = 0;
    
    while ((comma_pos = line.indexOf(',', start)) != -1 && field < 11) {
        String value = line.substring(start, comma_pos);
        
        switch (field) {
            case 0: sample.sample_id = value.toInt(); break;
            case 2: sample.scenario = value; break;
            case 3: sample.temp_sensor = value.toFloat(); break;
            case 4: sample.humidity_sensor = value.toFloat(); break;
            case 5: sample.mq135_raw = value.toFloat(); break;
            case 6: sample.mq135_normalized = value.toFloat(); break;
            case 7: sample.comfort_index = value.toFloat(); break;
            case 8: sample.temp_diff = value.toFloat(); break;
            case 9: sample.humidity_diff = value.toFloat(); break;
            case 10: sample.pm_ratio = value.toFloat(); break;
        }
        
        start = comma_pos + 1;
        field++;
    }
    
    // Last field
    if (field == 10) {
        sample.pm_ratio = line.substring(start).toFloat();
    }
    
    return sample;
}

void runPrediction(int index) {
    TestSample& sample = test_samples[index];
    
    Serial.printf("\n🤖 PREDICTION %d/%d\n", index + 1, test_samples.size());
    Serial.printf("🎯 Expected: %s\n", sample.scenario.c_str());
    
    // Prepare features
    float features[FEATURE_COUNT];
    features[0] = (sample.temp_sensor - SCALER_MEAN[0]) / SCALER_SCALE[0];
    features[1] = (sample.humidity_sensor - SCALER_MEAN[1]) / SCALER_SCALE[1];
    features[2] = (sample.mq135_raw - SCALER_MEAN[2]) / SCALER_SCALE[2];
    features[3] = (sample.mq135_normalized - SCALER_MEAN[3]) / SCALER_SCALE[3];
    features[4] = (sample.temp_diff - SCALER_MEAN[4]) / SCALER_SCALE[4];
    features[5] = (sample.humidity_diff - SCALER_MEAN[5]) / SCALER_SCALE[5];
    features[6] = (sample.comfort_index - SCALER_MEAN[6]) / SCALER_SCALE[6];
    features[7] = (sample.pm_ratio - SCALER_MEAN[7]) / SCALER_SCALE[7];
    
    // Run inference
    for (int i = 0; i < FEATURE_COUNT; i++) {
        input->data.f[i] = features[i];
    }
    
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("❌ Inference failed");
        return;
    }
    
    float predicted_aqi = output->data.f[0];
    predicted_aqi = constrain(predicted_aqi, 0, 500);
    
    // Display results
    Serial.printf("📊 Predicted AQI: %.1f\n", predicted_aqi);
    Serial.printf("📈 Level: %s\n", getAQILevel((int)predicted_aqi));
    Serial.printf("💡 Advice: %s\n", getAQIAdvice((int)predicted_aqi));
    
    // Send to Telegram (every 3rd prediction)
    if ((index + 1) % 3 == 0) {
        String emoji = getAQIEmoji((int)predicted_aqi);
        String message = emoji + " *ESP32 AI Prediction* " + emoji + "\n\n";
        message += "📋 Sample: " + String(index + 1) + "/" + String(test_samples.size()) + "\n";
        message += "🎯 Expected: " + sample.scenario + "\n\n";
        
        message += "📊 *Features:*\n";
        message += "🌡️ Temp: " + String(sample.temp_sensor, 1) + "°C\n";
        message += "💧 Humidity: " + String(sample.humidity_sensor, 1) + "%\n";
        message += "🌬️ Air Quality: " + String(sample.mq135_normalized, 1) + "%\n\n";
        
        message += "🤖 *AI Result:*\n";
        message += "📈 AQI: *" + String((int)predicted_aqi) + "*\n";
        message += "📊 Level: *" + String(getAQILevel((int)predicted_aqi)) + "*\n\n";
        
        message += "💡 " + String(getAQIAdvice((int)predicted_aqi)) + "\n\n";
        message += "⚡ *Embedded TensorFlow Lite*";
        
        sendTelegramMessage(message);
    }
}

void sendTelegramMessage(String message) {
    if (WiFi.status() != WL_CONNECTED) return;
    
    HTTPClient http;
    String url = "https://api.telegram.org/bot" + String(BOT_TOKEN) + "/sendMessage";
    
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    
    DynamicJsonDocument doc(2048);
    doc["chat_id"] = CHAT_ID;
    doc["text"] = message;
    doc["parse_mode"] = "Markdown";
    
    String jsonString;
    serializeJson(doc, jsonString);
    
    int response = http.POST(jsonString);
    if (response == 200) {
        Serial.println("✅ Telegram sent");
    } else {
        Serial.printf("❌ Telegram error: %d\n", response);
    }
    
    http.end();
}