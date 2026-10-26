"""
Unit tests for evaluation metrics correctness and edge-case handling.
"""
import pytest
import numpy as np
from src.evaluation.metrics import (
    calculate_wape,
    calculate_rmse,
    calculate_mae,
    calculate_bias,
    calculate_rmsse,
    compute_all_metrics
)


def test_wape_correctness():
    # actual = [10, 20], pred = [12, 18] -> errors = [2, 2], sum_err = 4, sum_act = 30 -> 4/30 = 0.1333
    actual = np.array([10.0, 20.0])
    pred = np.array([12.0, 18.0])
    wape = calculate_wape(actual, pred)
    assert pytest.approx(wape, 0.001) == 4.0 / 30.0


def test_wape_zero_denominator():
    # Both actuals and errors zero -> 0.0
    actual = np.array([0.0, 0.0])
    pred = np.array([0.0, 0.0])
    assert calculate_wape(actual, pred) == 0.0

    # Actuals zero but error > 0 -> inf
    pred_bad = np.array([5.0, 0.0])
    assert calculate_wape(actual, pred_bad) == float("inf")


def test_rmse_and_mae():
    actual = np.array([10.0, 20.0])
    pred = np.array([13.0, 16.0])
    # diffs = [3, -4] -> squared = [9, 16] -> mean = 12.5 -> sqrt = 3.5355
    assert pytest.approx(calculate_rmse(actual, pred), 0.001) == np.sqrt(12.5)
    # abs = [3, 4] -> mean = 3.5
    assert pytest.approx(calculate_mae(actual, pred), 0.001) == 3.5


def test_signed_bias_convention():
    actual = np.array([10.0, 20.0])
    # Over-forecasting: pred > actual -> positive bias
    pred_over = np.array([12.0, 24.0])
    assert calculate_bias(actual, pred_over) == 3.0  # mean([+2, +4])

    # Under-forecasting: pred < actual -> negative bias
    pred_under = np.array([8.0, 16.0])
    assert calculate_bias(actual, pred_under) == -3.0  # mean([-2, -4])


def test_rmsse_scaling():
    train_history = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0])
    actual = np.array([26.0, 28.0])
    pred = np.array([26.0, 28.0]) # Perfect forecast
    assert calculate_rmsse(actual, pred, train_history) == 0.0


def test_compute_all_metrics_clipping():
    actual = np.array([5.0, 10.0])
    pred = np.array([-2.0, 8.0]) # Negative prediction should be clipped to 0
    m = compute_all_metrics(actual, pred, clip_negative=True)
    # Clipped pred is [0, 8], errors are [5, 2] -> sum_err = 7, sum_act = 15 -> wape = 7/15 = 0.4667
    assert pytest.approx(m["wape"], 0.001) == 7.0 / 15.0
