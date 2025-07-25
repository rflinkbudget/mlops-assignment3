import os
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Original model
model = joblib.load("sklearn_model.joblib")
y_pred_original = model.predict(X_test)
r2_original = r2_score(y_test, y_pred_original)

# Load quantized model
quant_params = joblib.load("quant_params.joblib")
quantized_coef = quant_params["coef"].astype(np.float32)
coef_scale = quant_params["coef_scale"]
coef_min = quant_params["coef_min"]
dequant_intercept = quant_params["intercept"]

# Dequantize
dequant_coef = quantized_coef / coef_scale + coef_min

# Predict with dequantized weights
y_pred_quant = X_test @ dequant_coef + dequant_intercept
r2_quant = r2_score(y_test, y_pred_quant)

# File sizes
size_unquant = os.path.getsize("unquant_params.joblib") / 1024
size_quant = os.path.getsize("quant_params.joblib") / 1024

# Results
print("📊 Quantization Comparison Table:")
print(f"{'Metric':<20}{'Original Sklearn Model':<30}{'Quantized Model'}")
print(f"{'R² Score':<20}{r2_original:<30.4f}{r2_quant:.4f}")
print(f"{'Model Size (KB)':<20}{size_unquant:<30.2f}{size_quant:.2f}")
