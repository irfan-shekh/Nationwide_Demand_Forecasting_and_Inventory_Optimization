"""
Unit tests for feature engineering and strict temporal leakage prevention.
"""
import pytest
import pandas as pd
import numpy as np
from src.features.engineering import (
    add_calendar_features,
    add_price_features,
    add_historical_sales_features
)


def test_calendar_features_extraction():
    df = pd.DataFrame({
        "date": pd.date_range("2026-05-01", periods=7), # May 1, 2026 is Friday
        "event_name_1": ["None", "None", "Mother's Day", "None", "None", "None", "None"]
    })
    res = add_calendar_features(df)
    assert "dayofweek" in res.columns
    assert "is_weekend" in res.columns
    assert "is_event" in res.columns
    # Check weekend flag for Saturday and Sunday
    assert res.loc[1, "is_weekend"] == 1  # May 2 is Saturday
    assert res.loc[2, "is_event"] == 1    # Mother's Day


def test_strict_temporal_leakage_prevention():
    """
    CRITICAL LEAKAGE TEST:
    Modifying future sales at day t+k MUST NOT alter the engineered features at day t.
    """
    dates = pd.date_range("2026-01-01", periods=30)
    sales_clean = np.array([1.0] * 30)

    df1 = pd.DataFrame({
        "id": ["ITEM_A"] * 30,
        "date": dates,
        "sales": sales_clean
    })

    # Compute features for df1
    feat1 = add_historical_sales_features(df1, lags=[1, 7], rolling_windows=[7], min_lag_shift=1)

    # Now create df2 where day 25 is drastically corrupted (e.g. 1000 sales)
    df2 = df1.copy()
    df2.loc[24, "sales"] = 1000.0

    feat2 = add_historical_sales_features(df2, lags=[1, 7], rolling_windows=[7], min_lag_shift=1)

    # Features at day 20 (index 19) MUST be strictly identical
    assert feat1.loc[19, "sales_lag_1"] == feat2.loc[19, "sales_lag_1"]
    assert feat1.loc[19, "sales_roll_mean_7"] == feat2.loc[19, "sales_roll_mean_7"]

    # Even at day 25 (index 24), lag_1 and roll_mean_7 must NOT see day 25's 1000 sales!
    assert feat2.loc[24, "sales_lag_1"] == 1.0
    assert feat2.loc[24, "sales_roll_mean_7"] == 1.0  # Still 1.0 because shifted by 1


def test_direct_horizon_shift_leakage_prevention():
    """
    Verifies that when min_lag_shift=28, features at test day t do not even use sales
    from t-1 to t-27 (true direct multi-step forecasting).
    """
    dates = pd.date_range("2026-01-01", periods=60)
    df = pd.DataFrame({
        "id": ["ITEM_A"] * 60,
        "date": dates,
        "sales": np.arange(60, dtype=float)
    })

    feat = add_historical_sales_features(df, lags=[28], rolling_windows=[7], min_lag_shift=28)

    # At index 28 (day 29), lag_28 must be sales of day 1 (index 0) = 0.0
    assert feat.loc[28, "sales_lag_28"] == 0.0
    # At index 28, roll_mean_7 shifted by 28 must only see index 0
    assert feat.loc[28, "sales_roll_mean_7"] == 0.0
