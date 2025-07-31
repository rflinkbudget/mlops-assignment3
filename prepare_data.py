from sklearn.datasets import fetch_california_housing
import numpy as np, os

X, y = fetch_california_housing(return_X_y=True)
os.makedirs("data", exist_ok=True)
np.savez_compressed("data/california_housing.npz", X=X, y=y)
print("Saved to data/california_housing.npz")
