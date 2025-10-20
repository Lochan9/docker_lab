import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

if __name__ == "__main__":
    print("🚀 Starting model training...")

    # Load dataset
    df = pd.read_csv("winequality-red.csv", sep=";")

    # Separate features and labels
    X = df.drop("quality", axis=1).values
    y = df["quality"].values

    # Split into training/testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(11,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=8,
        verbose=1
    )

    # Evaluate
    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\n✅ Model trained successfully — MAE: {mae:.2f}")

    # Save model and scaler
    model.save("wine_model.keras")
    joblib.dump(scaler, "scaler.pkl")

    print("💾 Saved: wine_model.keras & scaler.pkl")
