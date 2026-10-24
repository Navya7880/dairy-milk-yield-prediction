from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data_processing import (
    CATEGORICAL_FEATURES,
    FEATURES,
    TARGET,
    ensure_synthetic_dataset,
    split_features_target,
    validate_dataset,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "synthetic_dairy_milk.csv"
MODEL_PATH = ROOT / "models" / "milk_yield_model.joblib"
METRICS_PATH = ROOT / "models" / "model_metadata.json"


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")),
         ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    preprocessor = ColumnTransformer(
        [("numeric", numeric_pipeline, FEATURES if not CATEGORICAL_FEATURES else [n for n in FEATURES if n not in CATEGORICAL_FEATURES]),
         ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)]
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])


def evaluate_and_train(data: pd.DataFrame, seed: int = 42) -> tuple[Pipeline, dict]:
    data = validate_dataset(data)
    features, target = split_features_target(data)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=seed)
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    baseline = np.full(len(y_test), y_train.mean())
    metrics = {
        "model_name": "Multiple Linear Regression",
        "dataset_type": "synthetic demonstration",
        "target": TARGET,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "seed": seed,
        "train_count": len(x_train),
        "test_count": len(x_test),
        "r2": round(float(r2_score(y_test, predicted)), 4),
        "mae": round(float(mean_absolute_error(y_test, predicted)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, predicted))), 4),
        "baseline_mae": round(float(mean_absolute_error(y_test, baseline)), 4),
        "baseline_rmse": round(float(np.sqrt(mean_squared_error(y_test, baseline))), 4),
        "actual": [round(float(v), 4) for v in y_test],
        "predicted": [round(float(v), 4) for v in predicted],
    }
    return pipeline, metrics


def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH, metadata_path: Path = METRICS_PATH) -> dict:
    data = pd.read_csv(data_path)
    pipeline, metrics = evaluate_and_train(data)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    ensure_synthetic_dataset(DATA_PATH)
    metrics = train()
    print(f"Saved model: {MODEL_PATH}")
    print(json.dumps({key: metrics[key] for key in ("train_count", "test_count", "r2", "mae", "rmse")}, indent=2))


if __name__ == "__main__":
    main()
