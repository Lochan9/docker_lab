from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import joblib
import os

# Define app and paths
BASE_DIR = os.path.dirname(__file__)
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

MODEL_PATH = os.path.join(BASE_DIR, "wine_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Load trained model & scaler
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Collect all inputs
            features = [
                float(request.form["fixed_acidity"]),
                float(request.form["volatile_acidity"]),
                float(request.form["citric_acid"]),
                float(request.form["residual_sugar"]),
                float(request.form["chlorides"]),
                float(request.form["free_sulfur_dioxide"]),
                float(request.form["total_sulfur_dioxide"]),
                float(request.form["density"]),
                float(request.form["ph"]),
                float(request.form["sulphates"]),
                float(request.form["alcohol"]),
            ]

            # Prepare for prediction
            input_data = np.array(features).reshape(1, -1)
            scaled_input = scaler.transform(input_data)

            prediction = model.predict(scaled_input)
            predicted_quality = float(np.clip(prediction[0][0], 0, 10))
            predicted_quality = round(predicted_quality, 2)

            return render_template("predict.html", prediction_text=f"Predicted Quality: {predicted_quality}")

        except Exception as e:
            return render_template("predict.html", prediction_text=f"Error: {str(e)}")

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
