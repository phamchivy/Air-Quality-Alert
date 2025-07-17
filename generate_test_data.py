import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducible results
np.random.seed(42)

def generate_test_data(num_samples=50):
    """
    Generate test data CSV for ESP32 without AQI labels
    Data will span different air quality conditions for comprehensive testing
    """
    
    test_data = []
    
    # Define different air quality scenarios
    scenarios = [
        # Scenario 1: Good air quality (AQI 0-50 expected)
        {
            'name': 'Good',
            'temp_range': (22, 28),
            'humidity_range': (45, 65),
            'mq135_range': (800, 1200),
            'samples': 15
        },
        # Scenario 2: Moderate air quality (AQI 51-100 expected)
        {
            'name': 'Moderate',
            'temp_range': (28, 32),
            'humidity_range': (35, 75),
            'mq135_range': (1200, 1800),
            'samples': 15
        },
        # Scenario 3: Unhealthy for sensitive (AQI 101-150 expected)
        {
            'name': 'Unhealthy_Sensitive',
            'temp_range': (32, 36),
            'humidity_range': (25, 45),
            'mq135_range': (1800, 2500),
            'samples': 10
        },
        # Scenario 4: Unhealthy (AQI 151-200 expected)
        {
            'name': 'Unhealthy',
            'temp_range': (35, 40),
            'humidity_range': (20, 40),
            'mq135_range': (2500, 3200),
            'samples': 7
        },
        # Scenario 5: Very unhealthy (AQI 201-300 expected)
        {
            'name': 'Very_Unhealthy',
            'temp_range': (38, 42),
            'humidity_range': (15, 35),
            'mq135_range': (3200, 3800),
            'samples': 3
        }
    ]
    
    sample_id = 1
    base_time = datetime.now()
    
    for scenario in scenarios:
        print(f"Generating {scenario['samples']} samples for: {scenario['name']}")
        
        for i in range(scenario['samples']):
            # Generate realistic sensor readings
            temp_sensor = np.random.uniform(*scenario['temp_range'])
            humidity_sensor = np.random.uniform(*scenario['humidity_range'])
            mq135_raw = int(np.random.uniform(*scenario['mq135_range']))
            
            # Add some realistic noise
            temp_sensor += np.random.normal(0, 0.5)  # ±0.5°C noise
            humidity_sensor += np.random.normal(0, 2.0)  # ±2% humidity noise
            mq135_raw += int(np.random.normal(0, 50))  # ±50 ADC noise
            
            # Ensure values are within realistic bounds
            temp_sensor = np.clip(temp_sensor, 15, 45)
            humidity_sensor = np.clip(humidity_sensor, 10, 95)
            mq135_raw = np.clip(mq135_raw, 0, 4095)
            
            # Calculate derived features (same as in training model)
            mq135_normalized = mq135_raw / 4095.0 * 100
            comfort_index = temp_sensor + humidity_sensor / 10
            
            # Create timestamp
            timestamp = base_time + timedelta(minutes=i*10)
            
            sample = {
                'sample_id': sample_id,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'scenario': scenario['name'],
                'temp_sensor': round(temp_sensor, 1),
                'humidity_sensor': round(humidity_sensor, 1),
                'mq135_raw': mq135_raw,
                'mq135_normalized': round(mq135_normalized, 2),
                'comfort_index': round(comfort_index, 2),
                'temp_diff': round(np.random.uniform(-3, 3), 1),  # Mock API difference
                'humidity_diff': round(np.random.uniform(-10, 10), 1),  # Mock API difference
                'pm_ratio': round(np.random.uniform(0.8, 1.5), 2)  # Mock PM ratio
            }
            
            test_data.append(sample)
            sample_id += 1
        
        # Update base time for next scenario
        base_time += timedelta(hours=2)
    
    return test_data

# Generate test data
print("🔬 Generating ESP32 test data...")
test_data = generate_test_data(50)

# Create DataFrame
df = pd.DataFrame(test_data)
print(f"\n✅ Generated {len(test_data)} test samples")

# Display summary by scenario
print("\n📊 Test Data Summary by Scenario:")
summary = df.groupby('scenario').agg({
    'temp_sensor': ['count', 'min', 'max', 'mean'],
    'humidity_sensor': ['min', 'max', 'mean'],
    'mq135_raw': ['min', 'max', 'mean'],
    'mq135_normalized': ['min', 'max', 'mean']
}).round(1)

print(summary)

# Reorder columns for better readability
columns_order = [
    'sample_id', 'timestamp', 'scenario',
    'temp_sensor', 'humidity_sensor', 'mq135_raw', 'mq135_normalized', 
    'comfort_index', 'temp_diff', 'humidity_diff', 'pm_ratio'
]

df_ordered = df[columns_order]

# Save as CSV
df_ordered.to_csv('data/esp32_test_data.csv', index=False)
print(f"\n💾 Saved as 'esp32_test_data.csv'")

# Display first few samples
print("\n🔍 First 10 test samples:")
print(df_ordered[['sample_id', 'scenario', 'temp_sensor', 'humidity_sensor', 'mq135_raw', 'mq135_normalized']].head(10))

print("\n🔍 Last 5 test samples:")
print(df_ordered[['sample_id', 'scenario', 'temp_sensor', 'humidity_sensor', 'mq135_raw', 'mq135_normalized']].tail(5))

# Show data distribution
print("\n📈 Data Distribution:")
print(f"Scenarios: {df['scenario'].value_counts().to_dict()}")

print("\n📋 Column Information:")
for col in df_ordered.columns:
    print(f"- {col}: {df_ordered[col].dtype}")

print("\n✅ Test data generation complete!")
print("\n📄 Generated file: esp32_test_data.csv")
print("📋 Contains all features needed for model prediction (except AQI)")
print("🎯 Ready for ESP32 model testing")

print("\n📊 CSV Structure:")
print("- sample_id: Unique identifier")
print("- timestamp: Test sample time") 
print("- scenario: Expected AQI level category")
print("- temp_sensor: ESP32 DHT11 temperature")
print("- humidity_sensor: ESP32 DHT11 humidity")
print("- mq135_raw: ESP32 MQ135 raw ADC value")
print("- mq135_normalized: Calculated percentage")
print("- comfort_index: Derived feature")
print("- temp_diff: Mock API temperature difference")
print("- humidity_diff: Mock API humidity difference") 
print("- pm_ratio: Mock PM10/PM2.5 ratio")

print(f"\n🎯 Model can predict AQI using these {len(columns_order)-3} features")