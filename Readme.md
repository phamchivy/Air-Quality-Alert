# ESP32 Air Quality Alert System

## Overview

This project implements an end-to-end air quality monitoring and alert system using an ESP32-S3 microcontroller and embedded AI. The system collects environmental data from local sensors, enriches it with online air quality data, trains a machine learning model on a computer, and deploys the model to the ESP32 for real-time inference and alerting via Telegram.

## Project Structure

```
Air-quality-alert/
│
├── data/                       # Raw and processed datasets
│   ├── esp32_test_data.csv
|   ├── filled_merged_data.csv
│   ├── sensor_data.csv
|   ├── waqi_data.csv
│
├── esp32_deployment/           # Generated files for ESP32 deployment
│   ├── esp32_aqi_model.tflite  # Trained TFLite model
│   ├── scaler_params.json      # Scaler parameters for normalization
│   ├── embedded_data.h         # Embedded model and test data as C++ arrays
│
├── esp32_aqi_inference/        # ESP32 Arduino source code
│   ├── esp32_aqi_inference.ino # Main ESP32 inference and alert code
│   ├── esp32_aqi_model.h       # Model parameters, scaler, and utility functions
|   ├── config.h                # Config Wifi and Telegram
│
├── pc_training.py              # Main Python training and export script
├── embed_files_method.py       # Script to embed model/data as C++ arrays
└── README.md                   # Project documentation
```

## Data Collection

- **Sensors Used:**
  - **DHT11:** Measures temperature and humidity.
  - **MQ135:** Measures air quality (gas sensor).
- **External Data:**
  - **WAQI API:** Fetches additional air quality data (e.g., PM2.5, PM10, AQI) from the World Air Quality Index.

## Data Processing

- Sensor data and WAQI API data are collected simultaneously.
- Some fields from the WAQI API may be missing; these are filled using correlation-based imputation.
- The final dataset combines local sensor readings and enriched WAQI data for model training.

## Model Training

- The training is performed on a computer using Python (TensorFlow/Keras and scikit-learn).
- Features are engineered and normalized.
- A lightweight neural network is trained to predict AQI based on sensor and API data.
- The trained model is converted to TensorFlow Lite (`.tflite`) format for deployment on ESP32.

## Deployment to ESP32

- The TFLite model and test data are embedded into C++ header files using the `embed_files_method.py` script.
- The ESP32 firmware (`esp32_aqi_inference.ino`) loads the embedded model and performs real-time inference on new sensor data.

## Real-Time Inference and Alerting

- The ESP32 connects to WiFi and collects sensor data.
- It normalizes the data using the same scaler parameters as in training.
- The embedded TFLite model predicts the AQI.
- The ESP32 sends air quality status and alerts to a Telegram bot, notifying users about current air quality and any warnings.

## Key Features

- **No SD card or SPIFFS required:** All model and test data are embedded in firmware.
- **End-to-end pipeline:** From data collection, enrichment, training, to deployment and real-time alerting.
- **Telegram integration:** Instant air quality alerts and advice sent to users.

## Getting Started

1. **Collect and preprocess data** using sensors and WAQI API (Remember to config token the in .env file).
2. **Train the model** on your computer with `pc_training.py`.
3. **Embed the model and data** using `embed_files_method.py`.
4. **Flash the ESP32** with the firmware in `esp32_aqi_inference/`.
5. **Configure WiFi and Telegram credentials** in `config.h`.
6. **Power up the ESP32** and receive real-time air quality alerts on Telegram.

---

**Author:**  
Team: Discord Football Team
Captain: Phạm Chí Vỹ - HUST
Member: Phùng Trọng Duy - HUMG