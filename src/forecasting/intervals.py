"""
Prediction Interval Utilities and Uncertainty Evaluation
Computes empirical interval coverage, average interval width,
and guarantees valid bounds (lower <= upper, non-negative demand).
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def evaluate_prediction_intervals(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float = 0.80
) -> Dict[str, float]:
    """
    Evaluates quality of prediction intervals:
    1. Empirical Coverage: proportion of actual values falling inside [lower, upper]
    2. Average Interval Width (Sharpness): mean(upper - lower)
    3. Coverage Error: empirical_coverage - nominal_coverage
    """
    act = np.asarray(actual, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    up = np.asarray(upper, dtype=np.float64)

    # Sanity clip: lower must not exceed upper
    low = np.minimum(low, up)
    low = np.clip(low, 0, None)
    up = np.clip(up, 0, None)

    mask = ~np.isnan(act) & ~np.isnan(low) & ~np.isnan(up)
    act = act[mask]
    low = low[mask]
    up = up[mask]

    if len(act) == 0:
        return {
            "nominal_coverage": nominal_coverage,
            "empirical_coverage": 0.0,
            "coverage_error": -nominal_coverage,
            "avg_width": 0.0,
            "sample_count": 0
        }

    within_bounds = (act >= low) & (act <= up)
    empirical_cov = float(np.mean(within_bounds))
    avg_width = float(np.mean(up - low))

    return {
        "nominal_coverage": round(nominal_coverage, 4),
        "empirical_coverage": round(empirical_cov, 4),
        "coverage_error": round(empirical_cov - nominal_coverage, 4),
        "avg_width": round(avg_width, 4),
        "sample_count": int(len(act))
    }


def compute_residual_prediction_intervals(
    residuals: np.ndarray,
    point_predictions: np.ndarray,
    confidence_level: float = 0.80
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs empirical residual-based prediction intervals (conformal/historical error approach).
    Suitable when a model outputs only point forecasts.
    """
    res = np.asarray(residuals, dtype=np.float64)
    preds = np.asarray(point_predictions, dtype=np.float64)

    alpha = 1.0 - confidence_level
    lower_perc = (alpha / 2.0) * 100
    upper_perc = (1.0 - alpha / 2.0) * 100

    q_low = np.percentile(res, lower_perc)
    q_high = np.percentile(res, upper_perc)

    lower_bounds = np.clip(preds + q_low, 0, None)
    upper_bounds = np.clip(preds + q_high, 0, None)

    # Ensure bounds order
    lower_bounds = np.minimum(lower_bounds, upper_bounds)

    return lower_bounds, upper_bounds
