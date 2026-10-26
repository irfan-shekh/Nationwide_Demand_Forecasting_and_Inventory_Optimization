"""
Baseline Forecasting Models
Mandatory benchmark baselines:
1. Last-Value (Naive 1-step persistence)
2. Seasonal-Naive (7-day weekly recurrence)
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class LastValueBaseline:
    """
    Carries the most recently observed actual sales value forward for all horizon steps.
    """
    def __init__(self):
        self.last_values_: Dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame, id_col: str = "id", target_col: str = "sales") -> "LastValueBaseline":
        # Extract the last observed sales value for each series
        sorted_df = train_df.sort_values(by="date")
        self.last_values_ = sorted_df.groupby(id_col)[target_col].last().to_dict()
        return self

    def predict(self, test_df: pd.DataFrame, id_col: str = "id") -> np.ndarray:
        preds = test_df[id_col].map(self.last_values_).fillna(0.0).values
        return np.clip(preds, 0, None)


class SeasonalNaiveBaseline:
    """
    Repeats the historical demand from the corresponding day of the previous week (lag-7 cycle).
    For horizon H > 7, cycles the last 7 observed days periodically.
    """
    def __init__(self, seasonal_period: int = 7):
        self.seasonal_period = seasonal_period
        self.last_week_values_: Dict[str, np.ndarray] = {}

    def fit(self, train_df: pd.DataFrame, id_col: str = "id", target_col: str = "sales") -> "SeasonalNaiveBaseline":
        sorted_df = train_df.sort_values(by="date")
        for series_id, grp in sorted_df.groupby(id_col):
            recent_sales = grp[target_col].values[-self.seasonal_period:]
            if len(recent_sales) < self.seasonal_period:
                # Pad with mean or zeros if series is shorter than 7 days
                padding = np.zeros(self.seasonal_period - len(recent_sales))
                recent_sales = np.concatenate([padding, recent_sales])
            self.last_week_values_[series_id] = recent_sales
        return self

    def predict(self, test_df: pd.DataFrame, id_col: str = "id", date_col: str = "date") -> np.ndarray:
        test_df = test_df.copy()
        # Ensure test dates are sorted per series
        preds = []
        for series_id, grp in test_df.groupby(id_col, sort=False):
            n_steps = len(grp)
            last_cycle = self.last_week_values_.get(series_id, np.zeros(self.seasonal_period))
            # Tile cycle across forecast horizon
            reps = int(np.ceil(n_steps / self.seasonal_period))
            tiled = np.tile(last_cycle, reps)[:n_steps]
            preds.extend(tiled)
        return np.clip(np.array(preds), 0, None)
