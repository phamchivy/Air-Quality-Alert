import requests
import csv
import time
from datetime import datetime
from dotenv import load_dotenv
import os

# --- Cấu hình ---
load_dotenv('.env')

API_TOKEN = os.environ.get('API_TOKEN_WAQI')        # <- Thay bằng token thật của bạn
LOCATION = "vietnam/ha-noi/chi-cuc-bvmt"                  # Hoặc dùng geo:21.0285;105.8542
DURATION_HOURS = 3                  # Thời gian chạy (giờ)
INTERVAL_SECONDS = 10              # Mỗi 15s 1 mẫu
OUTPUT_CSV = "data/waqi_data.csv"

# --- Tính số lần lấy mẫu ---
samples = int(DURATION_HOURS * 3600 / INTERVAL_SECONDS)

# --- Hàm lấy dữ liệu từ API WAQI ---
def fetch_air_quality():
    url = f"https://api.waqi.info/feed/{LOCATION}/?token={API_TOKEN}"
    try:
        response = requests.get(url, timeout=10).json()
        if response["status"] != "ok":
            return None

        data = response["data"]
        iaqi = data.get("iaqi", {})
        return {
            "timestamp": data["time"]["s"],
            "aqi": data.get("aqi"),
            "pm25": iaqi.get("pm25", {}).get("v"),
            "pm10": iaqi.get("pm10", {}).get("v"),
            "co": iaqi.get("co", {}).get("v"),
            "no2": iaqi.get("no2", {}).get("v"),
            "o3": iaqi.get("o3", {}).get("v"),
            "t": iaqi.get("t", {}).get("v"),
            "h": iaqi.get("h", {}).get("v"),
            "p": iaqi.get("p", {}).get("v"),
            "w": iaqi.get("w", {}).get("v"),
        }
    except Exception as e:
        print("Error:", e)
        return None

# --- Ghi file CSV ---
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "local_time", "timestamp", "aqi", "pm25", "pm10", "co", "no2", "o3", "t", "h", "p", "w"
    ])
    writer.writeheader()

    print(f"[START] Thu thập dữ liệu WAQI ({samples} mẫu mỗi {INTERVAL_SECONDS}s)")
    for i in range(samples):
        record = fetch_air_quality()
        if record:
            record["local_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow(record)
            f.flush() 
            print(f"[{i+1}/{samples}] OK: {record}")
        else:
            print(f"[{i+1}/{samples}] Lỗi, bỏ qua")

        time.sleep(INTERVAL_SECONDS)

print("[DONE] Ghi dữ liệu hoàn tất vào:", OUTPUT_CSV)
