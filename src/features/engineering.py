"""
Leakage-Safe Feature Engineering Pipeline
Constructs calendar, price, hierarchical, and historical lag/rolling features.
Enforces strict shift-before-rolling to prevent temporal information leakage.
"""
from typing import List, Optional
import pandas as pd
import numpy as np
import logging
from src.data.loader import reduce_mem_usage

logger = logging.getLogger(__name__)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts calendar and promotional features from date and event columns.
    Known future information valid across training and test horizons.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    df['dayofweek'] = df['date'].dt.dayofweek.astype(np.int8)
    df['day'] = df['date'].dt.day.astype(np.int8)
    df['month'] = df['date'].dt.month.astype(np.int8)
    df['year'] = df['date'].dt.year.astype(np.int16)
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(np.int8)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(np.int8)

    # Event indicator
    if 'event_name_1' in df.columns:
        df['is_event'] = (df['event_name_1'].fillna('None') != 'None').astype(np.int8)
    else:
        df['is_event'] = 0

    # SNAP matching state
    if 'state_id' in df.columns:
        snap_col = df['state_id'].map({
            'CA': 'snap_CA',
            'TX': 'snap_TX',
            'WI': 'snap_WI'
        })
        # If SNAP columns exist
        df['snap_active'] = 0
        for state, col in [('CA', 'snap_CA'), ('TX', 'snap_TX'), ('WI', 'snap_WI')]:
            if col in df.columns:
                mask = (df['state_id'] == state)
                df.loc[mask, 'snap_active'] = df.loc[mask, col].fillna(0).astype(np.int8)

    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes price changes and relative momentum features per store-item series.
    Shifted to ensure no future pricing leakage.
    """
    df = df.copy()
    if 'sell_price' not in df.columns:
        return df

    # Historical price statistics per item
    item_price_mean = df.groupby('item_id')['sell_price'].transform('mean')
    df['price_norm'] = (df['sell_price'] / item_price_mean.replace(0, np.nan)).fillna(1.0).astype(np.float32)

    # Week-over-week price change per series
    # sell_price changes weekly, shift by 1 day or compare with 7 days ago
    df['price_lag_7'] = df.groupby('id')['sell_price'].shift(7).astype(np.float32)
    df['price_diff_7'] = (df['sell_price'] - df['price_lag_7']).fillna(0.0).astype(np.float32)
    df['price_rel_diff_7'] = (df['price_diff_7'] / df['price_lag_7'].replace(0, np.nan)).fillna(0.0).astype(np.float32)

    return df


def add_historical_sales_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 7, 14, 28],
    rolling_windows: List[int] = [7, 14, 28],
    min_lag_shift: int = 1
) -> pd.DataFrame:
    """
    Computes lag and rolling sales features per series.
    CRITICAL LEAKAGE RULE:
    Every rolling feature is computed ONLY after shifting sales by at least min_lag_shift
    (e.g., shift(1) for 1-step ahead, or shift(28) for a 28-day direct forecast).
    This strictly guarantees that at prediction time t, no actual sales from t..t+H are observed.
    """
    df = df.copy()
    df = df.sort_values(by=['id', 'date']).reset_index(drop=True)

    grouped = df.groupby('id')['sales']

    # Lags
    for lag in lags:
        df[f'sales_lag_{lag}'] = grouped.shift(lag).astype(np.float32)

    # Rolling statistics shifted by min_lag_shift
    shifted_sales = grouped.shift(min_lag_shift)

    for w in rolling_windows:
        # Rolling Mean
        df[f'sales_roll_mean_{w}'] = shifted_sales.groupby(df['id']).rolling(w, min_periods=1).mean().reset_index(drop=True).astype(np.float32)
        # Rolling Std
        df[f'sales_roll_std_{w}'] = shifted_sales.groupby(df['id']).rolling(w, min_periods=1).std().reset_index(drop=True).fillna(0.0).astype(np.float32)
        # Rolling Min & Max
        df[f'sales_roll_min_{w}'] = shifted_sales.groupby(df['id']).rolling(w, min_periods=1).min().reset_index(drop=True).astype(np.float32)
        df[f'sales_roll_max_{w}'] = shifted_sales.groupby(df['id']).rolling(w, min_periods=1).max().reset_index(drop=True).astype(np.float32)

    return reduce_mem_usage(df)


def generate_all_features(
    df: pd.DataFrame,
    forecast_horizon: int = 28,
    direct_forecast: bool = False
) -> pd.DataFrame:
    """
    Orchestrates full feature generation pipeline.
    If direct_forecast is True, shifts all historical sales by forecast_horizon to enable
    direct H-step forecasting without recursive error propagation.
    """
    logger.info("Generating calendar features...")
    df = add_calendar_features(df)

    logger.info("Generating price features...")
    df = add_price_features(df)

    logger.info("Generating historical sales features...")
    min_shift = forecast_horizon if direct_forecast else 1
    lags = [forecast_horizon, forecast_horizon + 7, forecast_horizon + 14] if direct_forecast else [1, 7, 14, 28]

    df = add_historical_sales_features(
        df,
        lags=lags,
        rolling_windows=[7, 14, 28],
        min_lag_shift=min_shift
    )

    return df
