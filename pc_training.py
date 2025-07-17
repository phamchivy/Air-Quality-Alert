#!/usr/bin/env python3
"""
Complete PC Training Script for ESP32 Deployment
Trains model and generates all files needed for ESP32
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import json
import os

print("🚀 PC Training for ESP32 AQI Prediction")
print("="*50)

# --- STEP 1: Load and prepare training data ---
print("\n📊 STEP 1: Loading training data...")
try:
    data = pd.read_csv("data/filled_merged_data.csv")
    print(f"✅ Loaded {len(data)} samples")
    print(f"📋 Columns: {list(data.columns)}")
except FileNotFoundError:
    print("❌ filled_merged_data.csv not found!")
    print("Please ensure the file is in the current directory")
    exit(1)

# --- STEP 2: Feature engineering ---
print("\n🔧 STEP 2: Feature engineering...")
data['temp_diff'] = data['temp_sensor'] - data['t']
data['humidity_diff'] = data['humidity_sensor'] - data['h']
data['mq135_normalized'] = data['mq135_raw'] / 4095.0 * 100
data['pm_ratio'] = data['pm10'] / (data['pm25'] + 0.001)
data['comfort_index'] = data['temp_sensor'] + data['humidity_sensor'] / 10

# Define ESP32 features (must match order in ESP32 code!)
esp32_features = [
    'temp_sensor',      # 0
    'humidity_sensor',  # 1
    'mq135_raw',        # 2
    'mq135_normalized', # 3
    'temp_diff',        # 4
    'humidity_diff',    # 5
    'comfort_index',    # 6
    'pm_ratio'          # 7
]

print(f"📝 Features: {esp32_features}")

# Prepare data
X = data[esp32_features]
y = data['aqi']

# Remove missing values
mask = ~(X.isnull().any(axis=1) | y.isnull())
X = X[mask]
y = y[mask]

print(f"📊 Clean dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"📈 AQI range: {y.min():.1f} - {y.max():.1f}")

# --- STEP 3: Train/Test split ---
print("\n📋 STEP 3: Train/Test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"🏋️ Training: {X_train.shape[0]} samples")
print(f"🧪 Testing: {X_test.shape[0]} samples")

# --- STEP 4: Feature scaling ---
print("\n📏 STEP 4: Feature scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features normalized")
print(f"📊 Mean: {scaler.mean_}")
print(f"📊 Scale: {scaler.scale_}")

# --- STEP 5: Build TensorFlow model ---
print("\n🤖 STEP 5: Building TensorFlow model...")

# Lightweight model for ESP32
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(len(esp32_features),)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='linear')  # Regression output
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("📋 Model architecture:")
model.summary()

# --- STEP 6: Train model ---
print("\n🏋️ STEP 6: Training model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# --- STEP 7: Evaluate model ---
print("\n📊 STEP 7: Model evaluation...")
train_pred = model.predict(X_train_scaled).flatten()
test_pred = model.predict(X_test_scaled).flatten()

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"✅ Training MAE: {train_mae:.2f}, R²: {train_r2:.3f}")
print(f"✅ Test MAE: {test_mae:.2f}, R²: {test_r2:.3f}")

# --- STEP 8: Convert to TensorFlow Lite ---
print("\n🔄 STEP 8: Converting to TensorFlow Lite...")

# Convert with optimizations for ESP32
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Try different conversion strategies for ESP32 compatibility
conversion_success = False
tflite_model = None
model_size_kb = 0

# Strategy 1: Simple FLOAT32 conversion (most compatible with ESP32)
try:
    print("🔧 Trying FLOAT32 conversion (ESP32 compatible)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.allow_custom_ops = True
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    model_size_kb = len(tflite_model) / 1024
    print(f"✅ FLOAT32 conversion successful")
    print(f"📏 Model size: {model_size_kb:.1f} KB")
    conversion_success = True
    
except Exception as e:
    print(f"❌ FLOAT32 conversion failed: {e}")

# --- STEP 9: Save files for ESP32 ---
print("\n💾 STEP 9: Saving files for ESP32...")

# Create output directory
os.makedirs("esp32_deployment", exist_ok=True)

# 1. Save TFLite model
tflite_path = "esp32_deployment/esp32_aqi_model.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"✅ Saved: {tflite_path}")

# 2. Save scaler parameters
scaler_params = {
    'feature_names': esp32_features,
    'feature_count': len(esp32_features),
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'model_performance': {
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2)
    }
}

scaler_path = "esp32_deployment/scaler_params.json"
with open(scaler_path, 'w') as f:
    json.dump(scaler_params, f, indent=2)
print(f"✅ Saved: {scaler_path}")

# 3. Generate C++ header file for ESP32
cpp_header = f'''
// Auto-generated ESP32 AQI model parameters
// Generated from PC training script
// Model Performance: MAE={test_mae:.2f}, R²={test_r2:.3f}

#ifndef ESP32_AQI_MODEL_H
#define ESP32_AQI_MODEL_H

// Model configuration
const int FEATURE_COUNT = {len(esp32_features)};
const float MODEL_MAE = {test_mae:.2f}f;
const float MODEL_R2 = {test_r2:.3f}f;

// Feature names (for reference)
const char* FEATURE_NAMES[FEATURE_COUNT] = {{
    {', '.join([f'"{name}"' for name in esp32_features])}
}};

// Scaler parameters for feature normalization
const float SCALER_MEAN[FEATURE_COUNT] = {{
    {', '.join([f'{x:.6f}f' for x in scaler.mean_])}
}};

const float SCALER_SCALE[FEATURE_COUNT] = {{
    {', '.join([f'{x:.6f}f' for x in scaler.scale_])}
}};

// AQI level functions
const char* getAQILevel(int aqi) {{
    if (aqi <= 50) return "Good";
    else if (aqi <= 100) return "Moderate";
    else if (aqi <= 150) return "Unhealthy for Sensitive";
    else if (aqi <= 200) return "Unhealthy";
    else if (aqi <= 300) return "Very Unhealthy";
    else return "Hazardous";
}}

const char* getAQIAdvice(int aqi) {{
    if (aqi <= 50) return "Perfect for outdoor activities!";
    else if (aqi <= 100) return "Air quality is acceptable.";
    else if (aqi <= 150) return "Sensitive groups should reduce outdoor activities.";
    else if (aqi <= 200) return "Everyone should limit outdoor activities.";
    else if (aqi <= 300) return "Stay indoors. Use air purifier.";
    else return "Emergency conditions! Avoid all outdoor activities.";
}}

const char* getAQIEmoji(int aqi) {{
    if (aqi <= 50) return "🟢";
    else if (aqi <= 100) return "🟡";
    else if (aqi <= 150) return "🟠";
    else if (aqi <= 200) return "🔴";
    else if (aqi <= 300) return "🟣";
    else return "🚨";
}}

#endif // ESP32_AQI_MODEL_H
'''

header_path = "esp32_deployment/esp32_aqi_model.h"
with open(header_path, 'w', encoding='utf-8') as f:
    f.write(cpp_header)
print(f"✅ Saved: {header_path}")

# 4. Copy test data
test_data_path = "esp32_deployment/esp32_test_data.csv"
if os.path.exists("esp32_test_data.csv"):
    import shutil
    shutil.copy("esp32_test_data.csv", test_data_path)
    print(f"✅ Copied: {test_data_path}")
else:
    print("⚠️ esp32_test_data.csv not found - generate it first!")

# --- STEP 10: Test TFLite model ---
print("\n🧪 STEP 10: Testing TFLite model...")

# Load and test the TFLite model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"📋 TFLite input shape: {input_details[0]['shape']}")
print(f"📋 TFLite output shape: {output_details[0]['shape']}")

# Test with a sample
sample_input = X_test_scaled[0:1].astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], sample_input)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])

keras_output = model.predict(sample_input)

print(f"🔍 Sample test:")
print(f"   Keras prediction: {keras_output[0][0]:.1f}")
print(f"   TFLite prediction: {tflite_output[0][0]:.1f}")
print(f"   Difference: {abs(keras_output[0][0] - tflite_output[0][0]):.3f}")

# --- STEP 11: Generate deployment instructions ---
instructions = f'''
ESP32 DEPLOYMENT INSTRUCTIONS
=============================

FILES TO UPLOAD TO ESP32:
1. esp32_aqi_model.tflite ({model_size_kb:.1f} KB) - TensorFlow Lite model
2. esp32_test_data.csv - Test data without AQI labels
3. esp32_aqi_model.h - C++ header with parameters

ARDUINO LIBRARIES NEEDED:
- WiFi (built-in)
- HTTPClient (built-in)  
- ArduinoJson
- TensorFlowLite_ESP32

HARDWARE REQUIREMENTS:
- ESP32 with >= 4MB Flash
- microSD card module (optional, for CSV storage)
- WiFi connection for Telegram

MODEL PERFORMANCE:
- Training MAE: {train_mae:.2f}
- Test MAE: {test_mae:.2f}
- R² Score: {test_r2:.3f}
- Model Size: {model_size_kb:.1f} KB

NEXT STEPS:
1. Upload files to ESP32 via SD card or SPIFFS
2. Configure WiFi and Telegram credentials
3. Flash ESP32 inference code
4. ESP32 will load model and predict AQI for test data
5. Results sent to Telegram automatically
'''

instructions_path = "esp32_deployment/deployment_instructions.txt"
with open(instructions_path, 'w') as f:
    f.write(instructions)
print(f"✅ Saved: {instructions_path}")

print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"📁 All files saved in: esp32_deployment/")
print(f"📊 Model performance: MAE={test_mae:.2f}, R²={test_r2:.3f}")
print(f"📏 Model size: {model_size_kb:.1f} KB (suitable for ESP32)")
print("\n🚀 Ready for ESP32 deployment!")