import serial
import csv
from datetime import datetime
import time

# Cấu hình
ser = serial.Serial("COM6", 115200)  # Thay COM port
output_file = "data/sensor_data.csv"
samples = 1080

print("Đợi ESP32 khởi động và warm-up...")
time.sleep(35)  # Chờ 35s cho warm-up

# Đọc và bỏ qua setup messages
for _ in range(10):
    try:
        line = ser.readline().decode().strip()
        print(f"Setup: {line}")
        if "Starting data collection" in line:
            break
    except:
        pass

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "temperature", "humidity", "mq135_raw"])

    print(f"[START] Thu {samples} mẫu...")
    collected = 0
    
    while collected < samples:
        try:
            line = ser.readline().decode().strip()
            
            # Chỉ xử lý pure CSV data (3 values, 2 commas)
            if line.count(",") == 2 and not any(char.isalpha() for char in line):
                temp, hum, mq135 = map(float, line.split(","))
                
                # Validation
                if 0 <= temp <= 50 and 0 <= hum <= 100 and 0 <= mq135 <= 4095:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    writer.writerow([ts, temp, hum, mq135])
                    f.flush()
                    collected += 1
                    print(f"[{collected}/{samples}] {temp}°C, {hum}%, MQ={mq135}")
                    
        except Exception as e:
            continue

print(f"✅ Hoàn thành! Đã thu {collected} mẫu")
ser.close()