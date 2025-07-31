FROM python:3.9-slim

WORKDIR /app
COPY . .

# If you maintain requirements.txt, you can do:
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# Else:
RUN pip install --no-cache-dir scikit-learn joblib

# The image will contain sklearn_model.joblib if it was created before the build
# and predict.py will load it.
CMD ["python", "predict.py"]
