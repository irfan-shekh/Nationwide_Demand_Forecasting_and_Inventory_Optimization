"""
Unified Model Registry and Forecasting Orchestrator
Provides single interface for selecting, training, comparing, and predicting
across all supported models: Naive, Seasonal Naive, Prophet, LightGBM, and TFT.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from src.models.baselines import LastValueBaseline, SeasonalNaiveBaseline
from src.models.lightgbm_model import LightGBMForecaster
from src.models.prophet_model import batch_prophet_forecast
from src.models.deep_learning_model import TFTIntegrationManager
from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry maintaining trained models, performance metadata, and inference dispatch.
    """
    AVAILABLE_MODELS = [
        "Naive (Last-Value)",
        "Seasonal-Naive (7-Day)",
        "LightGBM",
        "Prophet",
        "Temporal Fusion Transformer (TFT)"
    ]

    def __init__(self):
        self.trained_models: Dict[str, Any] = {}
        self.metrics_history: Dict[str, Dict[str, float]] = {}
        self.tft_manager = TFTIntegrationManager()

    def register_model(self, name: str, model_obj: Any, metrics: Optional[Dict[str, float]] = None):
        self.trained_models[name] = model_obj
        if metrics:
            self.metrics_history[name] = metrics

    def get_comparison_table(self) -> pd.DataFrame:
        """Returns clean benchmark summary table of evaluated models."""
        rows = []
        for name, m in self.metrics_history.items():
            row = {"Model": name, **m}
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["Model", "sample_count", "wape", "rmse", "mae", "bias", "rmsse"])
        return pd.DataFrame(rows).sort_values(by="wape")

    def predict(
        self,
        model_name: str,
        test_df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Unified prediction dispatcher returning (forecast_df, metadata).
        forecast_df contains: ['id', 'date', 'sales', 'predicted', 'pred_lower', 'pred_upper']
        """
        res = test_df[['id', 'date', 'sales']].copy()
        provenance = {"model_name": model_name, "mode": "LIVE_INFERENCE"}

        if model_name == "Naive (Last-Value)":
            if "Naive (Last-Value)" in self.trained_models:
                m = self.trained_models["Naive (Last-Value)"]
            else:
                m = LastValueBaseline().fit(train_df)
            preds = m.predict(test_df)
            res['predicted'] = preds
            res['pred_lower'] = np.clip(preds * 0.7, 0, None)
            res['pred_upper'] = preds * 1.3

        elif model_name == "Seasonal-Naive (7-Day)":
            if "Seasonal-Naive (7-Day)" in self.trained_models:
                m = self.trained_models["Seasonal-Naive (7-Day)"]
            else:
                m = SeasonalNaiveBaseline().fit(train_df)
            preds = m.predict(test_df)
            res['predicted'] = preds
            res['pred_lower'] = np.clip(preds * 0.65, 0, None)
            res['pred_upper'] = preds * 1.35

        elif model_name == "LightGBM":
            if "LightGBM" not in self.trained_models:
                raise RuntimeError("LightGBM model has not been trained or loaded yet.")
            lgb_model: LightGBMForecaster = self.trained_models["LightGBM"]
            preds_dict = lgb_model.predict(test_df)
            res['predicted'] = preds_dict.get('point', np.zeros(len(test_df)))
            res['pred_lower'] = preds_dict.get('q_0.1', res['predicted'] * 0.75)
            res['pred_upper'] = preds_dict.get('q_0.9', res['predicted'] * 1.25)

        elif model_name == "Prophet":
            fc = batch_prophet_forecast(train_df, test_df)
            res = fc[['id', 'date', 'sales', 'predicted', 'pred_lower', 'pred_upper']]

        elif model_name == "Temporal Fusion Transformer (TFT)":
            # Attempt TFT manager load
            fallback = None
            if "LightGBM" in self.trained_models:
                fallback_preds = self.trained_models["LightGBM"].predict(test_df)
                fallback = test_df[['id', 'date', 'sales']].copy()
                fallback['predicted'] = fallback_preds['point']
                fallback['pred_lower'] = fallback_preds.get('q_0.1', fallback['predicted'] * 0.8)
                fallback['pred_upper'] = fallback_preds.get('q_0.9', fallback['predicted'] * 1.2)

            res, prov_desc = self.tft_manager.load_forecasts(test_df, fallback_df=fallback)
            provenance["provenance_description"] = prov_desc
            provenance["mode"] = self.tft_manager.mode

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Final non-negativity and bounds check
        res['predicted'] = np.clip(res['predicted'], 0, None)
        if 'pred_lower' in res.columns and 'pred_upper' in res.columns:
            res['pred_lower'] = np.clip(res['pred_lower'], 0, None)
            res['pred_upper'] = np.maximum(res['pred_lower'], res['pred_upper'])

        return res, provenance
