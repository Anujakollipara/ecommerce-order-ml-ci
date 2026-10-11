
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("ecommerce_model.pkl")

FEATURES = [
    "Gender",
    "Product_Category",
    "Quantity",
    "Price",
    "Payment_Method",
    "City",
    "Customer_Rating"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "ecommerce_model.pkl not found. "
            "Run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "ecommerce-order-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "A JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    sample = pd.DataFrame([{
        feature: data[feature]
        for feature in FEATURES
    }])

    model = load_model()
    prediction = str(model.predict(sample)[0])

    return jsonify({
        "predicted_order_status": prediction
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
