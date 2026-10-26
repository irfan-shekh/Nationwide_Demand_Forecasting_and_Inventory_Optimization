"""
Chronological Time-Series Backtesting and Validation Splitter
Guarantees clean time splits without temporal leakage or row shuffling.
"""
from typing import Tuple, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def chronological_split(
    df: pd.DataFrame,
    test_days: int = 28,
    val_days: int = 28,
    date_col: str = "date"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset chronologically into Train, Validation, and Test sets.
    - Test: last `test_days`
    - Validation: `val_days` immediately prior to Test
    - Train: all observations before Validation
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    max_date = df[date_col].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)

    train_df = df[df[date_col] < val_start].copy()
    val_df = df[(df[date_col] >= val_start) & (df[date_col] < test_start)].copy()
    test_df = df[df[date_col] >= test_start].copy()

    logger.info(
        f"Chronological split completed:\n"
        f"  Train: {train_df[date_col].min().date()} to {train_df[date_col].max().date()} ({len(train_df)} rows)\n"
        f"  Val:   {val_df[date_col].min().date()} to {val_df[date_col].max().date()} ({len(val_df)} rows)\n"
        f"  Test:  {test_df[date_col].min().date()} to {test_df[date_col].max().date()} ({len(test_df)} rows)"
    )
    return train_df, val_df, test_df


def create_rolling_splits(
    df: pd.DataFrame,
    num_folds: int = 3,
    horizon: int = 28,
    date_col: str = "date"
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generates expanding window rolling backtesting folds.
    """
    df = df.sort_values(by=date_col)
    max_date = df[date_col].max()
    folds = []

    for fold in range(num_folds):
        cutoff_test_end = max_date - pd.Timedelta(days=fold * horizon)
        cutoff_test_start = cutoff_test_end - pd.Timedelta(days=horizon - 1)

        train_part = df[df[date_col] < cutoff_test_start].copy()
        test_part = df[(df[date_col] >= cutoff_test_start) & (df[date_col] <= cutoff_test_end)].copy()
        if len(train_part) > 0 and len(test_part) > 0:
            folds.append((train_part, test_part))

    return folds[::-1]  # Return chronologically ordered folds
