import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True)
_, X_test, _, _ = train_test_split(X, y, test_size=0.2)
model = joblib.load("sklearn_model.joblib")
print("Sample prediction:", model.predict(X_test[:1]))
