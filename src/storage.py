from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SQLITE_PATH = ROOT / "data" / "dairy_records.db"
RECORD_COLUMNS = [
    "cattle_id", "breed", "age_months", "lactation_number", "body_weight_kg",
    "daily_feed_kg", "parity_stage", "days_in_milk", "milk_yield_liters", "record_date",
]


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "").strip()


def _sqlite_connection(path: str | Path = SQLITE_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def connection(path: str | Path = SQLITE_PATH) -> Iterator:
    url = database_url()
    if url:
        try:
            import psycopg
            conn = psycopg.connect(url)
        except ImportError as exc:
            raise RuntimeError("PostgreSQL support requires psycopg[binary].") from exc
    else:
        conn = _sqlite_connection(path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(path: str | Path = SQLITE_PATH) -> None:
    is_postgres = bool(database_url())
    sql = (
        """CREATE TABLE IF NOT EXISTS cattle_records (
        cattle_id TEXT PRIMARY KEY, breed TEXT NOT NULL, age_months REAL NOT NULL,
        lactation_number INTEGER NOT NULL, body_weight_kg REAL NOT NULL,
        daily_feed_kg REAL NOT NULL, parity_stage TEXT NOT NULL,
        days_in_milk INTEGER NOT NULL, milk_yield_liters REAL,
        record_date TEXT NOT NULL
        )"""
        if is_postgres
        else """CREATE TABLE IF NOT EXISTS cattle_records (
        cattle_id TEXT PRIMARY KEY, breed TEXT NOT NULL, age_months REAL NOT NULL,
        lactation_number INTEGER NOT NULL, body_weight_kg REAL NOT NULL,
        daily_feed_kg REAL NOT NULL, parity_stage TEXT NOT NULL,
        days_in_milk INTEGER NOT NULL, milk_yield_liters REAL,
        record_date TEXT NOT NULL
        )"""
    )
    with connection(path) as conn:
        conn.execute(sql)
        conn.commit()


def add_record(record: dict, path: str | Path = SQLITE_PATH) -> None:
    init_db(path)
    placeholders = ", ".join("?" if not database_url() else "%s" for _ in RECORD_COLUMNS)
    query = f"INSERT INTO cattle_records ({', '.join(RECORD_COLUMNS)}) VALUES ({placeholders})"
    values = tuple(record[column] for column in RECORD_COLUMNS)
    with connection(path) as conn:
        try:
            conn.execute(query, values)
            conn.commit()
        except Exception as exc:
            conn.rollback()
            if "UNIQUE" in str(exc).upper() or "PRIMARY" in str(exc).upper():
                raise ValueError(f"Cattle ID '{record['cattle_id']}' already exists.") from exc
            raise


def list_records(path: str | Path = SQLITE_PATH) -> pd.DataFrame:
    init_db(path)
    with connection(path) as conn:
        rows = conn.execute("SELECT * FROM cattle_records ORDER BY record_date DESC, cattle_id").fetchall()
    if not rows:
        return pd.DataFrame(columns=RECORD_COLUMNS)
    return pd.DataFrame([dict(row) for row in rows], columns=RECORD_COLUMNS)


def update_record(cattle_id: str, record: dict, path: str | Path = SQLITE_PATH) -> None:
    init_db(path)
    placeholder = "?" if not database_url() else "%s"
    columns = [column for column in RECORD_COLUMNS if column != "cattle_id"]
    assignments = ", ".join(f"{column} = {placeholder}" for column in columns)
    query = f"UPDATE cattle_records SET {assignments} WHERE cattle_id = {placeholder}"
    values = tuple(record[column] for column in columns) + (cattle_id,)
    with connection(path) as conn:
        cur = conn.execute(query, values)
        conn.commit()
        if cur.rowcount == 0:
            raise ValueError("Cattle record was not found.")


def delete_record(cattle_id: str, path: str | Path = SQLITE_PATH) -> None:
    init_db(path)
    placeholder = "?" if not database_url() else "%s"
    with connection(path) as conn:
        cur = conn.execute(f"DELETE FROM cattle_records WHERE cattle_id = {placeholder}", (cattle_id,))
        conn.commit()
        if cur.rowcount == 0:
            raise ValueError("Cattle record was not found.")
