"""
Unit tests for forecasting models (Naive, Seasonal-Naive, LightGBM, and prediction intervals).
"""
import pytest
import pandas as pd
import numpy as np
from src.models.baselines import LastValueBaseline, SeasonalNaiveBaseline
from src.forecasting.intervals import evaluate_prediction_intervals, compute_residual_prediction_intervals


def test_last_value_baseline():
    train_df = pd.DataFrame({
        "id": ["SERIES_1", "SERIES_1", "SERIES_2"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-02"]),
        "sales": [5.0, 9.0, 14.0]
    })
    test_df = pd.DataFrame({
        "id": ["SERIES_1", "SERIES_2", "SERIES_UNKNOWN"],
        "date": pd.to_datetime(["2026-01-03", "2026-01-03", "2026-01-03"])
    })

    baseline = LastValueBaseline().fit(train_df)
    preds = baseline.predict(test_df)
    assert preds[0] == 9.0   # Last of SERIES_1
    assert preds[1] == 14.0  # Last of SERIES_2
    assert preds[2] == 0.0   # Fallback for unknown series


def test_seasonal_naive_baseline_weekly_cycle():
    # 7-day pattern: 1, 2, 3, 4, 5, 6, 7
    cycle = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    train_df = pd.DataFrame({
        "id": ["SERIES_1"] * 7,
        "date": pd.date_range("2026-01-01", periods=7),
        "sales": cycle
    })
    # Predict 10 days ahead (should repeat 1..7, then 1, 2, 3)
    test_df = pd.DataFrame({
        "id": ["SERIES_1"] * 10,
        "date": pd.date_range("2026-01-08", periods=10)
    })

    snb = SeasonalNaiveBaseline(seasonal_period=7).fit(train_df)
    preds = snb.predict(test_df)
    assert len(preds) == 10
    assert list(preds[:7]) == cycle
    assert list(preds[7:]) == [1.0, 2.0, 3.0]


def test_prediction_intervals_bounds_validity():
    actual = np.array([5.0, 8.0, 12.0, 15.0])
    # Lower should be <= upper
    lower = np.array([4.0, 9.0, 10.0, 14.0])
    upper = np.array([7.0, 11.0, 13.0, 18.0])

    cov_res = evaluate_prediction_intervals(actual, lower, upper, nominal_coverage=0.80)
    assert cov_res["sample_count"] == 4
    assert 0.0 <= cov_res["empirical_coverage"] <= 1.0
    assert cov_res["avg_width"] >= 0.0


def test_residual_intervals_ordering():
    residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    preds = np.array([10.0, 20.0])
    lower, upper = compute_residual_prediction_intervals(residuals, preds, confidence_level=0.80)
    assert np.all(lower <= upper)
    assert np.all(lower >= 0)  # Non-negativity clipped
