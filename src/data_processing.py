from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TARGET = "milk_yield_liters"
FEATURES = [
    "breed",
    "age_months",
    "lactation_number",
    "body_weight_kg",
    "daily_feed_kg",
    "parity_stage",
    "days_in_milk",
]
CATEGORICAL_FEATURES = ["breed", "parity_stage"]
NUMERIC_FEATURES = [name for name in FEATURES if name not in CATEGORICAL_FEATURES]

LIMITS = {
    "age_months": (12, 180),
    "lactation_number": (1, 12),
    "body_weight_kg": (300, 850),
    "daily_feed_kg": (5, 35),
    "days_in_milk": (1, 305),
    "milk_yield_liters": (1, 50),
}


def generate_synthetic_data(rows: int = 800, seed: int = 42) -> pd.DataFrame:
    """Generate a clearly synthetic dairy-record demonstration dataset."""
    rng = np.random.default_rng(seed)
    breeds = np.array(["Holstein", "Jersey", "Brown Swiss", "Crossbred"])
    stages = np.array(["Early", "Mid", "Late"])
    breed = rng.choice(breeds, rows)
    stage = rng.choice(stages, rows)
    lactation = rng.integers(1, 6, rows)
    age = np.clip(32 + lactation * 24 + rng.normal(0, 8, rows), 28, 180)
    weight = np.clip(540 + (lactation - 1) * 18 + rng.normal(0, 46, rows), 300, 850)
    feed = np.clip(18 + 0.012 * (weight - 540) + rng.normal(0, 2.1, rows), 5, 35)
    dim = rng.integers(1, 306, rows)
    breed_effect = np.select(
        [breed == "Holstein", breed == "Jersey", breed == "Brown Swiss"],
        [2.2, -1.0, 0.8],
        default=0.3,
    )
    stage_effect = np.select([stage == "Early", stage == "Mid"], [3.0, 1.2], default=-1.5)
    curve = -0.00012 * (dim - 90) ** 2 + 0.025 * (305 - dim)
    yield_liters = (
        8.5
        + breed_effect
        + stage_effect
        + 0.003 * (weight - 540)
        + 0.55 * (feed - 18)
        + 1.1 * lactation
        + curve * 0.012
        + rng.normal(0, 1.25, rows)
    )
    yield_liters = np.clip(yield_liters, 1, 50)
    return pd.DataFrame(
        {
            "breed": breed,
            "age_months": age.round(1),
            "lactation_number": lactation,
            "body_weight_kg": weight.round(1),
            "daily_feed_kg": feed.round(2),
            "parity_stage": stage,
            "days_in_milk": dim,
            TARGET: yield_liters.round(2),
        }
    )


def ensure_synthetic_dataset(path: str | Path, rows: int = 800) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        generate_synthetic_data(rows).to_csv(path, index=False)
    return validate_dataset(pd.read_csv(path))


def validate_columns(data: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(data.columns))
    extra = sorted(set(data.columns) - set(required))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if extra:
        raise ValueError(f"Unexpected columns: {', '.join(extra)}")


def validate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    validate_columns(data, [*FEATURES, TARGET])
    if data.empty:
        raise ValueError("Dataset is empty.")
    numeric = NUMERIC_FEATURES + [TARGET]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].astype("string").str.strip()
    if data[numeric + CATEGORICAL_FEATURES].isna().any().any():
        raise ValueError("Dataset contains missing or non-numeric values.")
    for column, (minimum, maximum) in LIMITS.items():
        if not data[column].between(minimum, maximum).all():
            raise ValueError(f"Values in '{column}' must be between {minimum} and {maximum}.")
    if data.duplicated().any():
        raise ValueError("Dataset contains duplicate records.")
    return data.copy()


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    validate_columns(data, [*FEATURES, TARGET])
    return data.loc[:, FEATURES].copy(), data[TARGET].astype(float).copy()
