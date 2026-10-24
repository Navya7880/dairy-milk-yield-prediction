import pandas as pd
import pytest

from src.data_processing import FEATURES, TARGET, generate_synthetic_data, split_features_target, validate_dataset


def test_generated_dataset_shape_and_schema():
    data = generate_synthetic_data(40)
    assert data.shape == (40, len(FEATURES) + 1)
    assert set(FEATURES).issubset(data)
    assert TARGET in data
    assert validate_dataset(data).equals(data)


def test_validate_dataset_rejects_missing_and_duplicate_rows():
    data = generate_synthetic_data(20)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(data.drop(columns=[TARGET]))
    data.loc[1] = data.loc[0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_dataset(data)


def test_validate_dataset_rejects_out_of_range_target():
    data = generate_synthetic_data(20)
    data.loc[0, TARGET] = 100
    with pytest.raises(ValueError, match="milk_yield_liters"):
        validate_dataset(data)


def test_feature_target_separation():
    features, target = split_features_target(generate_synthetic_data(20))
    assert list(features.columns) == FEATURES
    assert target.name == TARGET
