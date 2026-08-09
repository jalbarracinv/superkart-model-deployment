from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]
NUMERIC_FEATURES = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Age_Years",
]

MODEL_PATH = Path(__file__).with_name("superkart_model.joblib")
model = joblib.load(MODEL_PATH)
superkart_api = Flask(__name__)


def prepare_input(frame):
    missing_columns = [column for column in FEATURES if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    prepared = frame.loc[:, FEATURES].copy()
    prepared["Product_Sugar_Content"] = prepared["Product_Sugar_Content"].replace(
        {"reg": "Regular"}
    )
    for column in NUMERIC_FEATURES:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    invalid_numeric = prepared[NUMERIC_FEATURES].isna().any()
    if invalid_numeric.any():
        invalid_columns = invalid_numeric[invalid_numeric].index.tolist()
        raise ValueError(f"Invalid numeric values in: {invalid_columns}")
    return prepared


@superkart_api.get("/")
@superkart_api.get("/health")
def health():
    return jsonify({"status": "ok", "model": "SuperKart sales regression"})


@superkart_api.post("/v1/predict")
def predict():
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON."}), 400
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body must be one object."}), 400
    try:
        input_data = prepare_input(pd.DataFrame([payload]))
        prediction = float(model.predict(input_data)[0])
        return jsonify({"prediction": prediction})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        superkart_api.logger.exception("Online prediction failed")
        return jsonify({"error": "Prediction could not be completed."}), 500


@superkart_api.post("/v1/predictbatch")
def predict_batch():
    if "file" not in request.files:
        return jsonify({"error": "A CSV file is required in the 'file' field."}), 400
    try:
        batch_data = pd.read_csv(request.files["file"])
        if batch_data.empty:
            raise ValueError("The uploaded CSV is empty.")
        input_data = prepare_input(batch_data)
        predictions = model.predict(input_data)
        result = {str(index): float(value) for index, value in enumerate(predictions)}
        return jsonify(result)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        superkart_api.logger.exception("Batch prediction failed")
        return jsonify({"error": "Batch prediction could not be completed."}), 500
