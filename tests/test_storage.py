from pathlib import Path

import pandas as pd
import pytest

from src.storage import add_record, delete_record, init_db, list_records, update_record


def record(cattle_id="COW-001"):
    return {
        "cattle_id": cattle_id, "breed": "Holstein", "age_months": 48,
        "lactation_number": 2, "body_weight_kg": 580, "daily_feed_kg": 18,
        "parity_stage": "Mid", "days_in_milk": 120, "milk_yield_liters": 20,
        "record_date": "2026-10-24",
    }


def test_database_crud_and_restart(tmp_path: Path):
    path = tmp_path / "records.db"
    init_db(path)
    assert list_records(path).empty
    add_record(record(), path)
    frame = list_records(path)
    assert frame.iloc[0]["cattle_id"] == "COW-001"
    changed = record()
    changed["milk_yield_liters"] = 22.5
    update_record("COW-001", changed, path)
    assert list_records(path).iloc[0]["milk_yield_liters"] == 22.5
    delete_record("COW-001", path)
    assert list_records(path).empty


def test_duplicate_cattle_id(tmp_path: Path):
    path = tmp_path / "records.db"
    add_record(record(), path)
    with pytest.raises(ValueError, match="already exists"):
        add_record(record(), path)


def test_update_and_delete_missing_record(tmp_path: Path):
    path = tmp_path / "records.db"
    with pytest.raises(ValueError, match="not found"):
        update_record("missing", record("missing"), path)
    with pytest.raises(ValueError, match="not found"):
        delete_record("missing", path)
