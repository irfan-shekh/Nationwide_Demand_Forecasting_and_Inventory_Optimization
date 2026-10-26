"""
Unit tests for data validation, schema checks, and memory reduction.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.validation import validate_schema, validate_sales_data, detect_intermittent_demand
from src.data.loader import reduce_mem_usage


def test_schema_validation_success():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=5),
        "item_id": ["ITEM_1"] * 5,
        "store_id": ["CA_1"] * 5,
        "sales": [0, 2, 4, 1, 0]
    })
    report = validate_schema(df, ["date", "item_id", "store_id", "sales"], "TestSchema")
    assert report.passed is True
    assert len(report.warnings) == 0
    assert report.duplicate_count == 0


def test_schema_validation_missing_columns():
    df = pd.DataFrame({"sales": [1, 2, 3]})
    report = validate_schema(df, ["date", "sales"], "MissingColTest")
    assert report.passed is False
    assert any("Missing required columns" in w for w in report.warnings)


def test_sales_validation_negative_check():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=3),
        "item_id": ["ITEM_1"] * 3,
        "store_id": ["CA_1"] * 3,
        "sales": [5, -2, 3]  # Negative sales anomaly
    })
    report = validate_sales_data(df)
    assert report.passed is False
    assert any("negative sales" in w for w in report.warnings)


def test_intermittent_demand_detection():
    # Mostly zeros
    intermittent_sales = pd.Series([0, 0, 0, 5, 0, 0, 0, 0, 4, 0])
    res = detect_intermittent_demand(intermittent_sales)
    assert res["zero_rate"] == 0.8
    assert res["is_intermittent"] is True

    # Continuous regular demand
    regular_sales = pd.Series([10, 12, 11, 10, 15, 14, 11, 13])
    res_reg = detect_intermittent_demand(regular_sales)
    assert res_reg["zero_rate"] == 0.0
    assert res_reg["is_intermittent"] is False


def test_reduce_mem_usage_downcasting():
    df = pd.DataFrame({
        "int_col": [1, 5, 10, 100],
        "float_col": [1.5, 2.5, 3.5, 4.5],
        "date": pd.date_range("2026-01-01", periods=4)
    })
    reduced = reduce_mem_usage(df)
    assert reduced["int_col"].dtype in [np.int8, np.int16]
    assert reduced["float_col"].dtype == np.float32
    assert pd.api.types.is_datetime64_any_dtype(reduced["date"])
