import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- Đọc file ---
print("Reading CSV files...")
api_df = pd.read_csv("data/waqi_data.csv")
sensor_df = pd.read_csv("data/sensor_data.csv")

print(f"API data shape: {api_df.shape}")
print(f"Sensor data shape: {sensor_df.shape}")

print("\nAPI columns:", list(api_df.columns))
print("Sensor columns:", list(sensor_df.columns))

# --- Kiểm tra missing values ---
print("\nAPI missing values:")
print(api_df.isnull().sum())

print("\nSensor missing values:")
print(sensor_df.isnull().sum())

# --- Ghép theo index (row by row) ---
api_df = api_df.reset_index(drop=True)
sensor_df = sensor_df.reset_index(drop=True)

min_rows = min(len(api_df), len(sensor_df))
print(f"\nUsing {min_rows} rows for merging")

# Trim to same length
api_trimmed = api_df.iloc[:min_rows].copy()
sensor_trimmed = sensor_df.iloc[:min_rows].copy()

# --- Merge by concatenating columns ---
# Bỏ timestamp columns để tránh duplicate
sensor_features = sensor_trimmed.drop(columns=['timestamp'], errors='ignore')

# Đổi tên cột sensor để tránh conflict
sensor_features = sensor_features.rename(columns={
    'temperature': 'temp_sensor',
    'humidity': 'humidity_sensor'
})

# Concatenate
merged = pd.concat([api_trimmed, sensor_features], axis=1)

# Bỏ timestamp columns
time_cols = ['local_time', 'timestamp']
for col in time_cols:
    if col in merged.columns:
        merged = merged.drop(columns=[col])

print(f"\nMerged data shape: {merged.shape}")
print("Final columns:", list(merged.columns))

# --- Analyze available data ---
target_columns = ['pm10', 'no2', 'o3']
available_data = {}

for col in target_columns:
    if col in merged.columns:
        non_null_count = merged[col].notna().sum()
        available_data[col] = non_null_count
        print(f"{col}: {non_null_count}/{len(merged)} values available")

# --- Strategy based on available data ---
if all(count == 0 for count in available_data.values()):
    print("\n❌ KHÔNG CÓ DATA NÀO CHO PM10, NO2, O3!")
    print("Sử dụng estimation models từ sensor data...")
    
    # Tạo estimated values từ sensor data
    result_data = merged.copy()
    
    # Simple estimation models dựa trên research
    # PM2.5 thường cao hơn PM10
    if 'pm25' in merged.columns:
        result_data['pm10'] = merged['pm25'] * 1.2  # PM10 ≈ 1.2 * PM2.5
    else:
        # Estimate từ MQ135 và temperature
        result_data['pm10'] = (merged['temp_sensor'] - 20) * 2 + merged['mq135_raw'] / 100
    
    # NO2 estimation từ CO và temperature
    if 'co' in merged.columns:
        result_data['no2'] = merged['co'] * 50 + merged['temp_sensor'] * 0.5
    else:
        result_data['no2'] = merged['temp_sensor'] * 1.5 + 10
    
    # O3 estimation (inverse relationship với humidity)
    result_data['o3'] = 100 - merged['humidity_sensor'] + merged['temp_sensor'] * 2
    
    print("✅ Created estimated values for missing targets")
    
else:
    print(f"\n✅ Found some target data: {available_data}")
    
    # --- Use RandomForest only if we have training data ---
    available_targets = [col for col, count in available_data.items() if count > 5]
    
    if not available_targets:
        print("⚠️ Too few samples for ML training, using simple estimation")
        result_data = merged.copy()
        
        # Simple estimation as fallback
        if 'pm25' in merged.columns:
            result_data['pm10'] = merged['pm25'] * 1.2
        if 'co' in merged.columns:
            result_data['no2'] = merged['co'] * 50
        result_data['o3'] = 100 - merged['humidity_sensor']
        
    else:
        print(f"Training ML models for: {available_targets}")
        
        # Prepare features (exclude target columns)
        feature_columns = [col for col in merged.columns if col not in target_columns]
        
        # Handle missing values in features
        features = merged[feature_columns].copy()
        
        # Fill missing features with simple strategies
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                # Fill numeric với mean of available values
                if features[col].notna().sum() > 0:
                    features[col] = features[col].fillna(features[col].mean())
                else:
                    features[col] = features[col].fillna(0)
            else:
                # Fill categorical với mode
                features[col] = features[col].fillna(features[col].mode()[0] if len(features[col].mode()) > 0 else 'unknown')
        
        print(f"Features after filling: {features.shape}")
        
        # Train models
        result_data = merged.copy()
        models = {}
        
        for col in available_targets:
            # Get training data
            train_mask = merged[col].notna()
            if train_mask.sum() >= 5:  # Need at least 5 samples
                X_train = features[train_mask]
                y_train = merged.loc[train_mask, col]
                
                print(f"Training {col} with {len(X_train)} samples")
                
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                models[col] = model
                
                # Predict missing values
                missing_mask = merged[col].isna()
                if missing_mask.sum() > 0:
                    predicted = model.predict(features[missing_mask])
                    result_data.loc[missing_mask, col] = predicted
                    print(f"Filled {missing_mask.sum()} missing values for {col}")

# --- Post-processing: Ensure realistic values ---
print("\nApplying realistic constraints...")

# Constrain values to realistic ranges
if 'pm10' in result_data.columns:
    result_data['pm10'] = np.clip(result_data['pm10'], 0, 500)

if 'no2' in result_data.columns:
    result_data['no2'] = np.clip(result_data['no2'], 0, 200)

if 'o3' in result_data.columns:
    result_data['o3'] = np.clip(result_data['o3'], 0, 300)

# --- Summary ---
print("\n=== FINAL SUMMARY ===")
print(f"Total rows: {len(result_data)}")
print(f"Total columns: {len(result_data.columns)}")

print("\nFinal missing values:")
for col in ['pm10', 'no2', 'o3']:
    if col in result_data.columns:
        missing_count = result_data[col].isnull().sum()
        print(f"{col}: {missing_count} missing ({missing_count/len(result_data)*100:.1f}%)")

print("\nValue ranges:")
for col in ['pm10', 'no2', 'o3']:
    if col in result_data.columns:
        print(f"{col}: {result_data[col].min():.1f} - {result_data[col].max():.1f}")

# --- Save result ---
output_file = "data/filled_merged_data.csv"
result_data.to_csv(output_file, index=False)

print(f"\n[✓] Saved to: {output_file}")

# Show sample
print("\nSample data:")
print(result_data.head())

print("\nColumn info:")
print(result_data.info())