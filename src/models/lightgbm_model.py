"""
LightGBM Global Machine Learning Demand Forecasting Model
Supports point forecasting (regression) and probabilistic quantile regression
for uncertainty intervals (e.g. 10th, 50th, 90th percentiles).
"""
from typing import List, Dict, Any, Optional, Tuple
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LightGBMForecaster:
    """
    Global LightGBM forecaster for retail time-series.
    Trained across multiple items, stores, and hierarchies simultaneously.
    """
    def __init__(
        self,
        features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        quantiles: Optional[List[float]] = None
    ):
        self.features = features or [
            'dayofweek', 'day', 'month', 'year', 'is_weekend', 'is_event', 'snap_active',
            'sell_price', 'price_diff_7', 'price_norm',
            'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
            'sales_roll_mean_7', 'sales_roll_std_7', 'sales_roll_min_7', 'sales_roll_max_7',
            'sales_roll_mean_14', 'sales_roll_std_14',
            'sales_roll_mean_28', 'sales_roll_std_28'
        ]
        self.categorical_features = categorical_features or [
            'store_id', 'dept_id', 'cat_id', 'state_id'
        ]
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42
        }
        self.quantiles = quantiles or [0.1, 0.5, 0.9]  # Default 80% prediction interval
        self.models: Dict[str, lgb.Booster] = {}
        self.categorical_mappings: Dict[str, Any] = {}

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepares features and converts categorical columns to category dtype."""
        data = df.copy()
        used_cats = []
        for cat in self.categorical_features:
            if cat in data.columns:
                data[cat] = data[cat].astype('category')
                used_cats.append(cat)

        available_features = [f for f in self.features if f in data.columns] + used_cats
        return data[available_features], used_cats

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        target_col: str = 'sales',
        num_boost_round: int = 400,
        early_stopping_rounds: int = 30
    ) -> "LightGBMForecaster":
        """
        Fits LightGBM point model and quantile models for uncertainty estimation.
        Uses chronological validation with early stopping.
        """
        X_train, cats = self._prepare_data(train_df)
        y_train = train_df[target_col].values

        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cats, free_raw_data=False)
        valid_sets = [dtrain]
        callbacks = [lgb.log_evaluation(period=0)]

        if val_df is not None and len(val_df) > 0:
            X_val, _ = self._prepare_data(val_df)
            y_val = val_df[target_col].values
            dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cats, reference=dtrain, free_raw_data=False)
            valid_sets.append(dval)
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

        # 1. Point model (RMSE / L2)
        logger.info("Training LightGBM point model...")
        point_params = self.params.copy()
        point_params['objective'] = 'regression'
        point_params['metric'] = 'rmse'

        self.models['point'] = lgb.train(
            point_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks
        )

        # 2. Quantile models for prediction intervals
        for q in self.quantiles:
            logger.info(f"Training LightGBM quantile model for alpha={q}...")
            q_params = self.params.copy()
            q_params['objective'] = 'quantile'
            q_params['alpha'] = q
            q_params['metric'] = 'quantile'

            # Quantile models train for slightly fewer rounds for speed
            self.models[f'q_{q}'] = lgb.train(
                q_params,
                dtrain,
                num_boost_round=min(num_boost_round, 200),
                valid_sets=valid_sets,
                callbacks=callbacks
            )

        return self

    def predict(self, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generates point forecasts and quantile intervals.
        Returns dictionary: {'point': array, 'q_0.1': array, 'q_0.9': array, ...}
        """
        X_test, _ = self._prepare_data(test_df)
        predictions = {}

        if 'point' in self.models:
            preds = self.models['point'].predict(X_test, num_iteration=self.models['point'].best_iteration)
            predictions['point'] = np.clip(preds, 0, None)

        for q in self.quantiles:
            key = f'q_{q}'
            if key in self.models:
                preds = self.models[key].predict(X_test, num_iteration=self.models[key].best_iteration)
                predictions[key] = np.clip(preds, 0, None)

        # Sanity check: Ensure lower quantile <= upper quantile
        if 'q_0.1' in predictions and 'q_0.9' in predictions:
            predictions['q_0.1'] = np.minimum(predictions['q_0.1'], predictions['q_0.9'])
            predictions['q_0.9'] = np.maximum(predictions['q_0.1'], predictions['q_0.9'])

        return predictions

    def save(self, filepath: Path) -> None:
        """Saves model dictionary to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved LightGBMForecaster to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "LightGBMForecaster":
        """Loads model from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)
