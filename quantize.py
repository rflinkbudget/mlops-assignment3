import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing


model = joblib.load("sklearn_model.joblib")
coef = model.coef_
intercept = model.intercept_

unquant_params = {"coef": coef, "intercept": intercept}
joblib.dump(unquant_params, "unquant_params.joblib")

scale = 255 / (np.max(coef) - np.min(coef))
quantized_coef = np.round((coef - np.min(coef)) * scale).astype(np.uint8)
quantized_intercept = np.round((intercept - np.min(coef)) * scale).astype(np.uint8)

quant_params = {"coef": quantized_coef, "intercept": quantized_intercept}
joblib.dump(quant_params, "quant_params.joblib")

# De-quantize and evaluate using PyTorch
dequant_coef = quantized_coef.astype(np.float32) / scale + np.min(coef)
dequant_intercept = quantized_intercept.astype(np.float32) / scale + np.min(coef)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(len(dequant_coef), 1)
        self.linear.weight = nn.Parameter(torch.tensor([dequant_coef], dtype=torch.float32))
        self.linear.bias = nn.Parameter(torch.tensor([dequant_intercept], dtype=torch.float32))

    def forward(self, x):
        return self.linear(x)

data = fetch_california_housing()
X_test = torch.tensor(data.data[:10], dtype=torch.float32)
model = SimpleModel()
print("Quantized model prediction:", model(X_test).detach().numpy())
