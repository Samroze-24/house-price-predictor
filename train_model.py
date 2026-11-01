"""Train the house-price regression model on the sample dataset.

Reads data/sample_houses.csv, trains a RandomForest regressor inside a
scikit-learn Pipeline (one-hot encoding for the categorical columns), and
evaluates it on a held-out 20% split. The printed R² and MAE are the real
results for this sample data -- they describe the model on synthetic data
only and are NOT a claim about real market accuracy.

Outputs:
    model.pkl    -- the fitted Pipeline (preprocessing + regressor)
    metrics.json -- R², MAE and the test-set size, for the web app to use

Usage:
    python train_model.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sample_houses.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

# Columns the model actually learns from. area_sqft is the canonical area
# (area_marla in the CSV is the same measurement, for human readability).
FEATURES = ["city", "area_sqft", "bedrooms", "bathrooms", "age_years", "location_tier"]
TARGET = "price_pkr"
CATEGORICAL = ["city", "location_tier"]
NUMERIC = ["area_sqft", "bedrooms", "bathrooms", "age_years"]


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH.name}")

    X = df[FEATURES]
    y = df[TARGET]

    # Held-out split so the metrics below are honest out-of-sample numbers.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Test set: {len(X_test)} houses")
    print(f"R²  = {r2:.4f}")
    print(f"MAE = PKR {mae:,.0f}")

    joblib.dump(model, MODEL_PATH)
    metrics = {"r2": round(float(r2), 4), "mae_pkr": round(float(mae), 2),
               "test_rows": int(len(X_test)), "train_rows": int(len(X_train))}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_PATH.name} and metrics to {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
