from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Simple Flask app to load the saved pipeline and serve predictions over HTTP.
# Save this as a new file (e.g., serve_model.py) and run it in the environment
# where model.pkl is present.


app = Flask(__name__)

# load the trained pipeline (predicts log(price))
model = joblib.load("models/model.pkl")

# expected feature columns in the same order used for training
FEATURES = ['area_type', 'location', 'total_sqft', 'bath', 'balcony', 'bhk']

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Model serving running. POST /predict with JSON payload."})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON payload:
    - single record: {"area_type": "...", "location": "...", "total_sqft": 1200, "bath": 2, "balcony": 1, "bhk": 2}
    - list of records: [ {...}, {...} ]
    Returns predicted log price and price (in lakhs) for each record.
    """
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "No JSON payload received"}), 400

    # normalize to list of records
    if isinstance(data, dict):
        records = [data]
    elif isinstance(data, list):
        records = data
    else:
        return jsonify({"error": "Payload must be a JSON object or list of objects"}), 400

    # create dataframe and ensure required columns exist
    df = pd.DataFrame(records)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # keep columns in same order as training
    X = df[FEATURES]

    try:
        pred_log = model.predict(X)  # pipeline predicts log(price)
        pred_price = np.exp(pred_log)  # convert back to price (same units as training target)
    except Exception as e:
        return jsonify({"error": "prediction failed", "details": str(e)}), 500

    results = []
    for i in range(len(pred_log)):
        results.append({
            "predicted_log_price": float(pred_log[i]),
            "predicted_price": float(pred_price[i])  # price in lakhs
        })

    return jsonify({"predictions": results})

if __name__ == "__main__":
    # for development use; in production use a WSGI server
    app.run(debug=True, port=5001)
