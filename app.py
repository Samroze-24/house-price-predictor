"""Flask web app: House Price Predictor (Pakistan-style sample data).

Routes:
    GET  /                    -- the predictor page
    GET  /api/city-averages   -- average sample price per city (from the CSV)
    POST /predict             -- JSON in, estimated price + explanation out

The price explanation is computed from the model itself, not hardcoded:
for each input we re-predict with that input replaced by a typical
reference value (the sample median / most common value). If the full
prediction is higher than the reference prediction, that input pushed
the price up, and vice versa.

IMPORTANT: predictions are based on a small synthetic sample dataset.
They are for demonstration only -- not a property valuation service
and not real market prices.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sample_houses.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

app = Flask(__name__)

CITIES = ["Lahore", "Karachi", "Islamabad", "Rawalpindi"]
TIERS = ["Premium", "Standard", "Basic"]
MARLA_SQFT = 272.25  # 1 marla = 272.25 sq ft (standard conversion)

# Loaded lazily on first use so `flask --app app` import stays light.
_model = None
_metrics = None
_sample = None


def get_model():
    """Load the trained pipeline, or raise a clear error if not trained yet."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "model.pkl not found. Run `python train_model.py` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def get_metrics():
    global _metrics
    if _metrics is None and METRICS_PATH.exists():
        _metrics = json.loads(METRICS_PATH.read_text())
    return _metrics or {}


def get_sample():
    global _sample
    if _sample is None:
        _sample = pd.read_csv(DATA_PATH)
    return _sample


def validate(payload):
    """Validate form/JSON input. Returns (clean_dict, error_message)."""
    try:
        city = str(payload.get("city", "")).strip()
        tier = str(payload.get("location_tier", "")).strip()
        area = float(payload.get("area", ""))
        unit = str(payload.get("area_unit", "marla")).strip().lower()
        bedrooms = int(payload.get("bedrooms", ""))
        bathrooms = int(payload.get("bathrooms", ""))
        age = int(payload.get("age_years", ""))
    except (TypeError, ValueError):
        return None, "All fields are required and must be numbers where expected."

    if city not in CITIES:
        return None, f"City must be one of: {', '.join(CITIES)}."
    if tier not in TIERS:
        return None, f"Location tier must be one of: {', '.join(TIERS)}."
    if unit not in ("marla", "sqft"):
        return None, "Area unit must be 'marla' or 'sqft'."
    if not (1 <= area <= 100):
        return None, "Area must be between 1 and 100 marla (or equivalent sq ft)."
    if not (1 <= bedrooms <= 10):
        return None, "Bedrooms must be between 1 and 10."
    if not (1 <= bathrooms <= 8):
        return None, "Bathrooms must be between 1 and 8."
    if not (0 <= age <= 60):
        return None, "Property age must be between 0 and 60 years."

    area_sqft = area * MARLA_SQFT if unit == "marla" else area
    if not (250 <= area_sqft <= 30000):
        return None, "Area must be between 250 and 30,000 sq ft."

    return {
        "city": city,
        "area_sqft": round(area_sqft, 2),
        "area_marla": round(area_sqft / MARLA_SQFT, 2),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
        "location_tier": tier,
    }, None


def predict_one(model, features):
    """Predict from a feature dict (keys must match the training columns)."""
    row = pd.DataFrame([{
        "city": features["city"],
        "area_sqft": features["area_sqft"],
        "bedrooms": features["bedrooms"],
        "bathrooms": features["bathrooms"],
        "age_years": features["age_years"],
        "location_tier": features["location_tier"],
    }])
    return float(model.predict(row)[0])


def explain(model, features, sample):
    """Model-computed impact of each input vs a typical reference house.

    Reference values come from the sample data itself (median for numbers,
    most common value for categories), so nothing here is hardcoded.
    """
    reference = {
        "city": sample["city"].mode()[0],
        "area_sqft": float(sample["area_sqft"].median()),
        "bedrooms": int(sample["bedrooms"].median()),
        "bathrooms": int(sample["bathrooms"].median()),
        "age_years": int(sample["age_years"].median()),
        "location_tier": sample["location_tier"].mode()[0],
    }
    base_prediction = predict_one(model, features)
    impacts = []
    labels = {
        "city": "City",
        "area_sqft": "Area",
        "bedrooms": "Bedrooms",
        "bathrooms": "Bathrooms",
        "age_years": "Property age",
        "location_tier": "Location tier",
    }
    for key, label in labels.items():
        tweaked = dict(features)
        tweaked[key] = reference[key]
        delta = base_prediction - predict_one(model, tweaked)
        impacts.append({"feature": label, "delta_pkr": round(delta, 2)})
    impacts.sort(key=lambda item: abs(item["delta_pkr"]), reverse=True)
    return impacts


@app.route("/")
def index():
    metrics = get_metrics()
    return render_template("index.html", cities=CITIES, tiers=TIERS, metrics=metrics)


@app.route("/api/city-averages")
def city_averages():
    """Average sample price per city, computed live from the CSV."""
    sample = get_sample()
    grouped = sample.groupby("city")["price_pkr"].agg(["mean", "count"])
    result = [
        {"city": city, "average_price_pkr": round(float(row["mean"]), 2),
         "sample_count": int(row["count"])}
        for city, row in grouped.iterrows()
    ]
    # Keep the order stable: highest average first.
    result.sort(key=lambda item: item["average_price_pkr"], reverse=True)
    return jsonify({"cities": result, "note": "Averages from the sample/synthetic dataset only."})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form
    features, error = validate(payload)
    if error:
        return jsonify({"success": False, "error": error}), 400

    try:
        model = get_model()
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

    sample = get_sample()
    price = predict_one(model, features)

    # Uncertainty band from the model's real test MAE, when available.
    mae = get_metrics().get("mae_pkr")
    half_band = float(mae) if mae else price * 0.10
    low, high = max(0, price - half_band), price + half_band

    # What-if: what does +1 bedroom do, according to the model?
    what_if = dict(features)
    what_if["bedrooms"] = min(10, features["bedrooms"] + 1)
    bedroom_effect = predict_one(model, what_if) - price

    return jsonify({
        "success": True,
        "estimated_price_pkr": round(price, 2),
        "price_range_pkr": {"low": round(low, 2), "high": round(high, 2)},
        "inputs": features,
        "feature_impacts": explain(model, features, sample),
        "what_if": {
            "change": "+1 bedroom",
            "new_price_pkr": round(price + bedroom_effect, 2),
            "delta_pkr": round(bedroom_effect, 2),
        },
        "model_metrics": get_metrics(),
        "disclaimer": "Estimate from a synthetic sample dataset. Not a valuation or market price.",
    })


if __name__ == "__main__":
    # Development server only; use a production WSGI server to deploy.
    app.run(host="127.0.0.1", port=5000, debug=True)
