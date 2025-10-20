# Stage 1: Train the model
FROM python:3.9 AS model_training
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_training.py .
COPY winequality-red.csv .
RUN python model_training.py

# Stage 2: Serve the Flask app
FROM python:3.9 AS serving
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=model_training /app/wine_model.keras .
COPY --from=model_training /app/scaler.pkl .

COPY main.py .
COPY templates ./templates

EXPOSE 4000
CMD ["python", "main.py"]
