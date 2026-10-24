from pathlib import Path

import joblib
import pytest

from src.data_processing import FEATURES, generate_synthetic_data
from src.prediction import PredictionError, load_model, predict_yield, validate_prediction_input
from src.train_model import evaluate_and_train


def valid_values():
    return dict(zip(FEATURES, ["Holstein", 48, 2, 580, 18, "Mid", 120]))


def test_training_prediction_save_reload(tmp_path: Path):
    pipeline, metrics = evaluate_and_train(generate_synthetic_data(240))
    assert metrics["test_count"] > 0
    assert metrics["r2"] >= 0
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    loaded = load_model(model_path)
    result = predict_yield(model_path, valid_values())
    assert isinstance(result, float)
    assert result > 0
    assert loaded.predict(validate_prediction_input(valid_values())).shape == (1,)


def test_missing_extra_and_invalid_prediction_inputs():
    values = valid_values()
    values.pop(FEATURES[0])
    with pytest.raises(PredictionError, match="Missing inputs"):
        validate_prediction_input(values)
    values[FEATURES[0]] = "Holstein"
    values["unknown"] = 1
    with pytest.raises(PredictionError, match="Unexpected inputs"):
        validate_prediction_input(values)
    values.pop("unknown")
    values["age_months"] = -1
    with pytest.raises(PredictionError, match="age_months"):
        validate_prediction_input(values)


def test_missing_model_has_recovery_message(tmp_path: Path):
    with pytest.raises(PredictionError, match="Run the training script"):
        load_model(tmp_path / "missing.joblib")
