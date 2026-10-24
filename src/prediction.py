from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .data_processing import CATEGORICAL_FEATURES, FEATURES, LIMITS


class PredictionError(ValueError):
    pass


def load_model(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise PredictionError(f"Model file not found: {path.name}. Run the training script first.")
    try:
        model = joblib.load(path)
    except (OSError, EOFError) as exc:
        raise PredictionError("The saved model file is missing or incompatible.") from exc
    if not hasattr(model, "predict"):
        raise PredictionError("The saved model does not provide predictions.")
    return model


def model_schema(model) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    if not hasattr(model, "feature_names_in_"):
        raise PredictionError("The saved model does not declare expected input features.")
    features = [str(name) for name in model.feature_names_in_.tolist()]
    categorical: dict[str, list[str]] = {}
    types: dict[str, str] = {}
    try:
        preprocessor = model.named_steps["preprocessor"]
        encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        for name, values in zip(preprocessor.transformers_[1][2], encoder.categories_):
            categorical[str(name)] = [str(value) for value in values]
        types.update({name: "category" for name in categorical})
    except (AttributeError, KeyError, IndexError) as exc:
        raise PredictionError("The saved model has an incompatible preprocessing schema.") from exc
    types.update({name: "number" for name in features if name not in categorical})
    return features, categorical, types


def load_prediction_schema(path: str | Path) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    return model_schema(load_model(path))


def validate_prediction_input(values: dict[str, Any], model=None) -> pd.DataFrame:
    features = FEATURES
    categorical = {name: [] for name in CATEGORICAL_FEATURES}
    if model is not None:
        features, categorical, _ = model_schema(model)
    missing = [name for name in features if name not in values]
    extra = [name for name in values if name not in features]
    if missing:
        raise PredictionError(f"Missing inputs: {', '.join(missing)}")
    if extra:
        raise PredictionError(f"Unexpected inputs: {', '.join(extra)}")
    row: dict[str, Any] = {}
    for name in features:
        value = values[name]
        if name in categorical:
            text = str(value).strip()
            if categorical.get(name) and text not in categorical[name]:
                raise PredictionError(f"{name} must be one of: {', '.join(categorical[name])}.")
            row[name] = text
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise PredictionError(f"{name} must be numeric.") from exc
        minimum, maximum = LIMITS[name]
        if not minimum <= number <= maximum:
            raise PredictionError(f"{name} must be between {minimum} and {maximum}.")
        row[name] = number
    return pd.DataFrame([row], columns=features)


def predict_yield(model_path: str | Path, values: dict[str, Any]) -> float:
    model = load_model(model_path)
    frame = validate_prediction_input(values, model)
    try:
        prediction = float(model.predict(frame)[0])
    except Exception as exc:
        raise PredictionError("The model rejected these inputs. Check the selected values.") from exc
    if prediction < 0:
        raise PredictionError("The model returned an invalid negative yield.")
    return round(prediction, 2)
