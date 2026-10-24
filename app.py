
from pathlib import Path
import logging
import sys
import time
import uuid
import joblib
import pandas as pd
from flask import Flask, jsonify, request,g

app = Flask(__name__)
# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger("ecommerce-ml-api")


# Log every HTTP request and its response
@app.before_request
def log_request_start():
    g.request_id = request.headers.get(
        "X-Request-ID", str(uuid.uuid4())
    )
    g.start_time = time.perf_counter()


@app.after_request
def log_request_end(response):
    request_id = getattr(g, "request_id", "unknown")
    start_time = getattr(g, "start_time", time.perf_counter())

    duration_ms = (time.perf_counter() - start_time) * 1000

    level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        level,
        "event=http_request request_id=%s method=%s "
        "path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    response.headers["X-Request-ID"] = request_id
    return response

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
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "ecommerce_model.pkl not found. "
                "Run train_model.py first."
            )

        model = joblib.load(MODEL_PATH)
        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH.name
        )
        return model

    except Exception:
        logger.exception(
            "event=model_load_failed model_file=%s",
            MODEL_PATH.name
        )
        raise


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "ecommerce-order-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    # Check whether the request contains a JSON object
    if not isinstance(data, dict):
        logger.warning(
            "event=invalid_json request_id=%s",
            g.request_id
        )
        return jsonify({
            "error": "A JSON request body is required"
        }), 400

    # Check required fields
    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=missing_fields request_id=%s fields=%s",
            g.request_id,
            missing_fields
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        logger.info(
            "event=prediction_request request_id=%s "
            "feature_names=%s",
            g.request_id,
            sorted(FEATURES)
        )

        model = load_model()
        prediction = str(model.predict(sample)[0])

        logger.info(
            "event=prediction_success request_id=%s",
            g.request_id
        )

        return jsonify({
            "predicted_order_status": prediction
        })

    except Exception:
        logger.exception(
            "event=prediction_failed request_id=%s",
            g.request_id
        )
        return jsonify({
            "error": "Prediction failed"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
