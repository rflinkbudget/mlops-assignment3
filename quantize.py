import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing

# Load sklearn model
model = joblib.load("sklearn_model.joblib")
coef = model.coef_
intercept = model.intercept_  # float

# Save unquantized params
unquant_params = {"coef": coef, "intercept": intercept}
joblib.dump(unquant_params, "unquant_params.joblib")

# Quantize only coef
coef_min, coef_max = np.min(coef), np.max(coef)
coef_scale = 255 / (coef_max - coef_min)
quantized_coef = np.round((coef - coef_min) * coef_scale).astype(np.uint8)

# Save quantized model
quant_params = {
    "coef": quantized_coef,
    "coef_scale": coef_scale,
    "coef_min": coef_min,
    "intercept": intercept  # original, not quantized
}
joblib.dump(quant_params, "quant_params.joblib")

# Dequantize for test
dequant_coef = quantized_coef.astype(np.float32) / coef_scale + coef_min
dequant_intercept = intercept

# Simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(len(dequant_coef), 1)
        self.linear.weight = nn.Parameter(torch.tensor([dequant_coef], dtype=torch.float32))
        self.linear.bias = nn.Parameter(torch.tensor([dequant_intercept], dtype=torch.float32))

    def forward(self, x):
        return self.linear(x)

# Test inference
data = fetch_california_housing()
X_test = torch.tensor(data.data[:10], dtype=torch.float32)
model = SimpleModel()
print("Quantized model prediction:", model(X_test).detach().numpy())
