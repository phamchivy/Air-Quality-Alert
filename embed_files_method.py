# Script để convert files thành C++ arrays
# Chạy script này sau khi train xong để tạo embedded data

import os

def file_to_cpp_array(file_path, array_name):
    """Convert file to C++ byte array"""
    with open(file_path, 'rb') as f:
        data = f.read()
    
    cpp_code = f"// Auto-generated from {file_path}\n"
    cpp_code += f"const unsigned char {array_name}[] PROGMEM = {{\n"
    
    for i, byte in enumerate(data):
        if i % 16 == 0:
            cpp_code += "    "
        cpp_code += f"0x{byte:02x}"
        if i < len(data) - 1:
            cpp_code += ", "
        if i % 16 == 15 or i == len(data) - 1:
            cpp_code += "\n"
    
    cpp_code += f"}};\n"
    cpp_code += f"const unsigned int {array_name}_len = {len(data)};\n\n"
    
    return cpp_code

def csv_to_cpp_string(csv_path):
    """Convert CSV to C++ string literal"""
    with open(csv_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Escape quotes and newlines
    content = content.replace('\\', '\\\\')
    content = content.replace('"', '\\"')
    lines = content.split('\n')
    
    cpp_code = "// Auto-generated CSV data\n"
    cpp_code += "const char* embedded_csv_data = \n"
    
    for i, line in enumerate(lines):
        if line.strip():  # Skip empty lines
            cpp_code += f'    "{line}\\n"\n'
    
    cpp_code += ";\n\n"
    
    return cpp_code

print("🔄 Converting files to embedded C++ code...")

# Check if files exist
model_file = "esp32_deployment/esp32_aqi_model.tflite"
csv_file = "data/esp32_test_data.csv"

if not os.path.exists(model_file):
    print(f"❌ {model_file} not found! Run training script first.")
    exit(1)

if not os.path.exists(csv_file):
    print(f"❌ {csv_file} not found!")
    exit(1)

# Generate embedded files
cpp_output = ""

# 1. Embed TensorFlow Lite model
print("📦 Embedding TensorFlow Lite model...")
cpp_output += file_to_cpp_array(model_file, "embedded_model_data")

# 2. Embed CSV data  
print("📦 Embedding CSV test data...")
cpp_output += csv_to_cpp_string(csv_file)

# 3. Create complete header file
header_content = f'''
#ifndef EMBEDDED_DATA_H
#define EMBEDDED_DATA_H

#include <Arduino.h>

{cpp_output}

// Helper functions
bool loadEmbeddedModel() {{
    // Model data is in embedded_model_data[]
    // Size is embedded_model_data_len
    return true;
}}

String getEmbeddedCSV() {{
    return String(embedded_csv_data);
}}

#endif // EMBEDDED_DATA_H
'''

# Save header file
output_file = "esp32_deployment/embedded_data.h"
with open(output_file, 'w') as f:
    f.write(header_content)

print(f"✅ Generated: {output_file}")

# Get file sizes
model_size = os.path.getsize(model_file) / 1024
csv_size = os.path.getsize(csv_file) / 1024

print(f"\n📊 File sizes:")
print(f"Model: {model_size:.1f} KB")
print(f"CSV: {csv_size:.1f} KB")
print(f"Total: {model_size + csv_size:.1f} KB")

if model_size + csv_size > 100:
    print("\n⚠️  WARNING: Total size > 100KB")
    print("💡 Consider using SPIFFS instead of embedding")
else:
    print("\n✅ Size OK for embedding in code")

print(f"\n🎯 Next steps:")
print(f"1. Include embedded_data.h in your ESP32 sketch")
print(f"2. Use embedded_model_data[] for TensorFlow Lite")
print(f"3. Use getEmbeddedCSV() for test data")
print(f"4. No need for SD card or SPIFFS!")