"""
Evaluation Metrics Module for Time-Series Retail Demand Forecasting
Implements WAPE, RMSE, MAE, Signed Bias, and Scaled Error (RMSSE).
Handles zero-denominators, missing values, and hierarchical aggregations safely.
"""
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import json


def calculate_wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (WAPE / MAD/Mean ratio):
    sum(|actual - predicted|) / sum(|actual|)
    Returns 0.0 if both sum of actuals and absolute error are 0.
    Returns inf if actuals sum to 0 but error > 0.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    abs_errors = np.abs(actual - predicted)
    sum_actual = np.sum(np.abs(actual))
    if sum_actual == 0:
        return 0.0 if np.sum(abs_errors) == 0 else float("inf")
    return float(np.sum(abs_errors) / sum_actual)


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error (MAE)."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return float(np.mean(np.abs(actual - predicted)))


def calculate_bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Signed Mean Forecast Error (Bias):
    mean(predicted - actual)
    Positive value indicates over-forecasting (excess inventory risk).
    Negative value indicates under-forecasting (stockout risk).
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return float(np.mean(predicted - actual))


def calculate_rmsse(
    actual: np.ndarray,
    predicted: np.ndarray,
    historical_train: np.ndarray,
    seasonal_lag: int = 7
) -> float:
    """
    Root Mean Squared Scaled Error (RMSSE):
    M5 competition primary scale-independent metric.
    Denominator is in-sample mean squared error of a seasonal naive forecast.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    historical_train = np.asarray(historical_train, dtype=np.float64)

    h_mse = np.mean((actual - predicted) ** 2)

    if len(historical_train) <= seasonal_lag:
        # Fallback to lag-1 diff
        diffs = np.diff(historical_train)
    else:
        diffs = historical_train[seasonal_lag:] - historical_train[:-seasonal_lag]

    scale = np.mean(diffs ** 2) if len(diffs) > 0 else 1.0
    if scale == 0:
        return float(np.sqrt(h_mse)) if h_mse > 0 else 0.0

    return float(np.sqrt(h_mse / scale))


def compute_all_metrics(
    actual: Union[pd.Series, np.ndarray],
    predicted: Union[pd.Series, np.ndarray],
    historical_train: Optional[Union[pd.Series, np.ndarray]] = None,
    clip_negative: bool = True
) -> Dict[str, float]:
    """
    Computes standard suite of demand forecasting metrics on aligned series.
    """
    act = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)

    # Valid mask (no NaNs)
    mask = ~np.isnan(act) & ~np.isnan(pred)
    act = act[mask]
    pred = pred[mask]

    if clip_negative:
        pred = np.clip(pred, 0, None)

    if len(act) == 0:
        return {
            "sample_count": 0,
            "wape": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "bias": 0.0,
            "rmsse": 0.0
        }

    res = {
        "sample_count": int(len(act)),
        "wape": round(calculate_wape(act, pred), 4),
        "rmse": round(calculate_rmse(act, pred), 4),
        "mae": round(calculate_mae(act, pred), 4),
        "bias": round(calculate_bias(act, pred), 4),
    }

    if historical_train is not None and len(historical_train) > 0:
        res["rmsse"] = round(calculate_rmsse(act, pred, np.asarray(historical_train)), 4)

    return res


def evaluate_hierarchical_performance(
    df: pd.DataFrame,
    levels: Optional[list] = None
) -> Dict[str, pd.DataFrame]:
    """
    Evaluates forecasting metrics across all hierarchy levels:
    National, State, Store, Department, Category, Item.
    Expects df to have columns: ['actual', 'predicted', 'date', 'state_id', 'store_id', 'dept_id', 'cat_id', 'item_id'].
    """
    target_levels = levels or ['state_id', 'store_id', 'dept_id', 'cat_id']
    results = {}

    # National Total Level
    nat_metrics = compute_all_metrics(df['actual'], df['predicted'])
    results['National'] = pd.DataFrame([{"level": "National", "entity": "All_Stores", **nat_metrics}])

    for lvl in target_levels:
        if lvl in df.columns:
            lvl_records = []
            for name, grp in df.groupby(lvl):
                m = compute_all_metrics(grp['actual'], grp['predicted'])
                lvl_records.append({"level": lvl, "entity": str(name), **m})
            results[lvl] = pd.DataFrame(lvl_records)

    return results
