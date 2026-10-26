"""
Prophet Forecasting Model Implementation
Provides additive/multiplicative trend and seasonality modeling with holiday integration
and built-in Monte Carlo uncertainty intervals.
"""
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet package is not installed. ProphetForecaster will run in fallback mode.")


class ProphetForecaster:
    """
    Fits Facebook/Meta Prophet on a retail time-series.
    Generates point forecast (yhat) and prediction intervals (yhat_lower, yhat_upper).
    """
    def __init__(
        self,
        interval_width: float = 0.95,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False
    ):
        self.interval_width = interval_width
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None

    def fit(self, df: pd.DataFrame, date_col: str = "date", target_col: str = "sales") -> "ProphetForecaster":
        """
        Fits Prophet model on a single series DataFrame with date and sales.
        """
        if not PROPHET_AVAILABLE:
            raise RuntimeError("Prophet is not available in the current environment.")

        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df[date_col]),
            'y': df[target_col].astype(float)
        }).sort_values('ds')

        self.model = Prophet(
            interval_width=self.interval_width,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )
        self.model.add_country_holidays(country_name='US')
        # Suppress verbose stan logs
        self.model.fit(prophet_df)
        return self

    def predict(self, horizon_days: int = 28) -> pd.DataFrame:
        """
        Generates forecast for future horizon_days.
        Returns DataFrame with ['ds', 'yhat', 'yhat_lower', 'yhat_upper'].
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")

        future = self.model.make_future_dataframe(periods=horizon_days, include_history=False)
        forecast = self.model.predict(future)
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        result['yhat'] = np.clip(result['yhat'], 0, None)
        result['yhat_lower'] = np.clip(result['yhat_lower'], 0, None)
        result['yhat_upper'] = np.clip(result['yhat_upper'], 0, None)
        return result


def batch_prophet_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str = "id",
    interval_width: float = 0.95
) -> pd.DataFrame:
    """
    Fits and forecasts multiple series using Prophet.
    Returns aligned predictions with intervals.
    """
    if not PROPHET_AVAILABLE:
        # Fallback to seasonal-naive with simple empirical intervals
        from src.models.baselines import SeasonalNaiveBaseline
        snb = SeasonalNaiveBaseline().fit(train_df)
        preds = snb.predict(test_df)
        res = test_df[[group_col, 'date', 'sales']].copy()
        res['predicted'] = preds
        res['pred_lower'] = np.clip(preds * 0.7, 0, None)
        res['pred_upper'] = preds * 1.3
        return res

    results = []
    horizon = len(test_df['date'].unique())

    for series_id, train_group in train_df.groupby(group_col):
        test_group = test_df[test_df[group_col] == series_id].sort_values('date')
        if len(test_group) == 0:
            continue
        try:
            forecaster = ProphetForecaster(interval_width=interval_width)
            forecaster.fit(train_group)
            fc = forecaster.predict(horizon_days=len(test_group))

            merged = test_group.copy().reset_index(drop=True)
            merged['predicted'] = fc['yhat'].values[:len(merged)]
            merged['pred_lower'] = fc['yhat_lower'].values[:len(merged)]
            merged['pred_upper'] = fc['yhat_upper'].values[:len(merged)]
            results.append(merged)
        except Exception as e:
            logger.warning(f"Prophet failed for series {series_id}: {e}. Using seasonal-naive fallback.")
            recent = train_group.sort_values('date')['sales'].values[-7:]
            reps = int(np.ceil(len(test_group) / 7))
            tiled = np.tile(recent, reps)[:len(test_group)]
            fallback_df = test_group.copy().reset_index(drop=True)
            fallback_df['predicted'] = tiled
            fallback_df['pred_lower'] = np.clip(tiled * 0.7, 0, None)
            fallback_df['pred_upper'] = tiled * 1.3
            results.append(fallback_df)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
