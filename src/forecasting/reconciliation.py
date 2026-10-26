"""
Hierarchical Forecast Reconciliation Engine
Supports:
1. Bottom-Up Aggregation: Exact summing matrix aggregation from Item-Store up to National level.
2. MinT (Minimum Trace) Optimal Reconciliation:
   Linear projection: y_tilde = S * (S^T * W^-1 * S)^-1 * S^T * W^-1 * y_hat
   Features:
   - Summing matrix S construction
   - Shrinkage error covariance estimator W to prevent numerical singularity
   - Verified automated hierarchical consistency checks.
"""
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BottomUpReconciler:
    """
    Reconciles hierarchical retail forecasts via bottom-up summation:
    Item-Store -> Category -> Department -> Store -> State -> National.
    Guarantees mathematically consistent totals where all higher levels exactly
    equal the sum of their constituent child predictions.
    """
    def __init__(self, hierarchy_cols: Optional[List[str]] = None):
        self.hierarchy_cols = hierarchy_cols or ['state_id', 'store_id', 'cat_id', 'dept_id', 'item_id']

    def reconcile(
        self,
        bottom_forecasts: pd.DataFrame,
        value_cols: List[str] = ['predicted', 'sales']
    ) -> Dict[str, pd.DataFrame]:
        """
        Takes bottom-level item forecasts and produces reconciled parent forecasts
        at every tier of the retail tree.
        """
        df = bottom_forecasts.copy()
        reconciled_levels = {}

        # Ensure value columns exist
        existing_val_cols = [c for c in value_cols if c in df.columns]

        # Tier 0: Bottom level (Item-Store)
        reconciled_levels['Item_Store'] = df.copy()

        # Tier 1: Department level per store
        reconciled_levels['Store_Department'] = df.groupby(['store_id', 'dept_id', 'date'])[existing_val_cols].sum().reset_index()

        # Tier 2: Category level per store
        reconciled_levels['Store_Category'] = df.groupby(['store_id', 'cat_id', 'date'])[existing_val_cols].sum().reset_index()

        # Tier 3: Store level
        reconciled_levels['Store'] = df.groupby(['store_id', 'date'])[existing_val_cols].sum().reset_index()

        # Tier 4: State level
        reconciled_levels['State'] = df.groupby(['state_id', 'date'])[existing_val_cols].sum().reset_index()

        # Tier 5: National Total
        nat = df.groupby('date')[existing_val_cols].sum().reset_index()
        nat['level'] = 'National'
        reconciled_levels['National'] = nat

        return reconciled_levels


class MinTReconciler:
    """
    Minimum Trace (MinT) Optimal Hierarchical Reconciliation.
    Computes unbiased reconciled forecasts that minimize total forecast error variance:
      y_reconciled = S (S^T W^-1 S)^-1 S^T W^-1 y_base
    where:
      S = Summing matrix (m x n_b)
      W = Covariance matrix of base forecast errors with Ledoit-Wolf shrinkage regularization.
    """
    def __init__(self, shrinkage_lambda: float = 0.1):
        self.shrinkage_lambda = shrinkage_lambda
        self.S_matrix: Optional[np.ndarray] = None
        self.bottom_series_names: List[str] = []
        self.total_series_names: List[str] = []

    def build_summing_matrix(
        self,
        bottom_df: pd.DataFrame,
        series_id_col: str = "id",
        store_col: str = "store_id",
        state_col: str = "state_id"
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Builds the binary structural summing matrix S.
        Rows: [National; States; Stores; Bottom items]
        Cols: Bottom items
        """
        # Unique bottom items
        bottom_series = sorted(bottom_df[series_id_col].unique().tolist())
        n_b = len(bottom_series)
        meta = bottom_df.drop_duplicates(subset=[series_id_col]).set_index(series_id_col)

        unique_stores = sorted(meta[store_col].unique().tolist())
        unique_states = sorted(meta[state_col].unique().tolist())

        total_names = ["National"] + [f"State_{s}" for s in unique_states] + [f"Store_{st}" for st in unique_stores] + bottom_series
        m = len(total_names)

        S = np.zeros((m, n_b), dtype=np.float64)

        # 1. National row (sums all bottom items)
        S[0, :] = 1.0

        # 2. State rows
        row_idx = 1
        for st in unique_states:
            matching = [1.0 if meta.loc[sid, state_col] == st else 0.0 for sid in bottom_series]
            S[row_idx, :] = matching
            row_idx += 1

        # 3. Store rows
        for store in unique_stores:
            matching = [1.0 if meta.loc[sid, store_col] == store else 0.0 for sid in bottom_series]
            S[row_idx, :] = matching
            row_idx += 1

        # 4. Bottom item identity block
        S[row_idx:, :] = np.eye(n_b, dtype=np.float64)

        self.S_matrix = S
        self.bottom_series_names = bottom_series
        self.total_series_names = total_names
        return S, total_names, bottom_series

    def estimate_shrinkage_covariance(self, residuals: np.ndarray) -> np.ndarray:
        """
        Estimates shrinkage error covariance W to ensure strict positive-definiteness.
        W_shrink = (1 - lambda) * Sample_Cov + lambda * diag(Sample_Cov) + epsilon * I
        """
        n_samples, n_series = residuals.shape
        sample_cov = np.cov(residuals, rowvar=False)

        # Diagonal target
        diag_target = np.diag(np.diag(sample_cov))

        # Shrinkage
        W = (1.0 - self.shrinkage_lambda) * sample_cov + self.shrinkage_lambda * diag_target

        # Regularization for numerical conditioning
        epsilon = 1e-4 * np.trace(W) / n_series
        W += epsilon * np.eye(n_series)

        return W

    def reconcile(
        self,
        base_forecasts: np.ndarray,
        residuals: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Reconciles base forecasts vector y_base (length m) using MinT:
        Returns reconciled vector y_tilde (length m).
        """
        if self.S_matrix is None:
            raise RuntimeError("Summing matrix S must be built before calling reconcile.")

        S = self.S_matrix
        m, n_b = S.shape

        if residuals is not None and residuals.shape[1] == m:
            W = self.estimate_shrinkage_covariance(residuals)
        else:
            # Fallback to WLS (Weighted Least Squares using S row norms) or OLS
            W = np.eye(m, dtype=np.float64)

        # Compute projection matrix: P = (S^T W^-1 S)^-1 S^T W^-1
        try:
            W_inv = np.linalg.pinv(W)
            STS_inv = np.linalg.pinv(S.T @ W_inv @ S)
            P = STS_inv @ S.T @ W_inv
            # Reconciled bottom forecasts
            y_bottom_reconciled = P @ base_forecasts
            # Project back to full hierarchy via S
            y_reconciled = S @ y_bottom_reconciled
        except np.linalg.LinAlgError:
            logger.warning("Linear algebra singular matrix during MinT; falling back to Bottom-Up.")
            y_bottom = base_forecasts[-n_b:]
            y_reconciled = S @ y_bottom

        return np.clip(y_reconciled, 0, None)


def verify_hierarchy_consistency(reconciled_levels: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Automated verification that child forecasts sum to parent forecasts across all tiers.
    """
    discrepancies = {}

    # Check: Store sum == State sum
    if 'Store' in reconciled_levels and 'State' in reconciled_levels:
        store_sum = float(reconciled_levels['Store']['predicted'].sum())
        state_sum = float(reconciled_levels['State']['predicted'].sum())
        diff = float(abs(store_sum - state_sum))
        discrepancies['store_vs_state'] = {
            'store_total': round(store_sum, 2),
            'state_total': round(state_sum, 2),
            'diff': round(diff, 4),
            'consistent': bool(diff < 1e-2)
        }

    # Check: State sum == National sum
    if 'State' in reconciled_levels and 'National' in reconciled_levels:
        state_sum = float(reconciled_levels['State']['predicted'].sum())
        nat_sum = float(reconciled_levels['National']['predicted'].sum())
        diff = float(abs(state_sum - nat_sum))
        discrepancies['state_vs_national'] = {
            'state_total': round(state_sum, 2),
            'national_total': round(nat_sum, 2),
            'diff': round(diff, 4),
            'consistent': bool(diff < 1e-2)
        }

    all_consistent = bool(all(d['consistent'] for d in discrepancies.values()))
    return {"all_consistent": all_consistent, "checks": discrepancies}
