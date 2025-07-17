import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import json
import os

interpreter = tf.lite.Interpreter(model_path="esp32_deployment/esp32_aqi_model.tflite")
interpreter.allocate_tensors()
for detail in interpreter.get_tensor_details():
    print(detail['name'], detail['dtype'])
