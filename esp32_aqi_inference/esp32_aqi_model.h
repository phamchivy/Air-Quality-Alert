
// Auto-generated ESP32 AQI model parameters
// Generated from PC training script
// Model Performance: MAE=3.08, R²=0.947

#ifndef ESP32_AQI_MODEL_H
#define ESP32_AQI_MODEL_H

// Model configuration
const int FEATURE_COUNT = 8;
const float MODEL_MAE = 3.08f;
const float MODEL_R2 = 0.947f;

// Feature names (for reference)
const char* FEATURE_NAMES[FEATURE_COUNT] = {
    "temp_sensor", "humidity_sensor", "mq135_raw", "mq135_normalized", "temp_diff", "humidity_diff", "comfort_index", "pm_ratio"
};

// Scaler parameters for feature normalization
const float SCALER_MEAN[FEATURE_COUNT] = {
    33.510069f, 68.575116f, 807.055556f, 19.708316f, 0.389699f, -5.028472f, 40.367581f, 1.199979f
};

const float SCALER_SCALE[FEATURE_COUNT] = {
    0.129736f, 0.285764f, 126.357450f, 3.085652f, 0.645514f, 2.505013f, 0.114095f, 0.000005f
};

// AQI level functions
const char* getAQILevel(int aqi) {
    if (aqi <= 50) return "Good";
    else if (aqi <= 100) return "Moderate";
    else if (aqi <= 150) return "Unhealthy for Sensitive";
    else if (aqi <= 200) return "Unhealthy";
    else if (aqi <= 300) return "Very Unhealthy";
    else return "Hazardous";
}

const char* getAQIAdvice(int aqi) {
    if (aqi <= 50) return "Perfect for outdoor activities!";
    else if (aqi <= 100) return "Air quality is acceptable.";
    else if (aqi <= 150) return "Sensitive groups should reduce outdoor activities.";
    else if (aqi <= 200) return "Everyone should limit outdoor activities.";
    else if (aqi <= 300) return "Stay indoors. Use air purifier.";
    else return "Emergency conditions! Avoid all outdoor activities.";
}

const char* getAQIEmoji(int aqi) {
    if (aqi <= 50) return "🟢";
    else if (aqi <= 100) return "🟡";
    else if (aqi <= 150) return "🟠";
    else if (aqi <= 200) return "🔴";
    else if (aqi <= 300) return "🟣";
    else return "🚨";
}

#endif // ESP32_AQI_MODEL_H
