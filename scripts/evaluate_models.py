"""
Hierarchical Evaluation and Reconciliation Verification Script
Evaluates multi-level retail hierarchy metrics and tests MinT vs Bottom-Up aggregation.
"""
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import SAMPLE_DATA_DIR, REPORTS_DIR
from src.evaluation.metrics import evaluate_hierarchical_performance, compute_all_metrics
from src.forecasting.reconciliation import BottomUpReconciler, MinTReconciler, verify_hierarchy_consistency

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== STEP 1: Loading Test Predictions ===")
    pred_path = SAMPLE_DATA_DIR / "test_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found at {pred_path}. Run train_models.py first.")

    df = pd.read_parquet(pred_path)
    df['predicted'] = df['pred_lightgbm']  # Use best local production model

    logger.info("=== STEP 2: Multi-Level Retail Hierarchy Performance ===")
    hier_results = evaluate_hierarchical_performance(df, levels=['state_id', 'store_id', 'dept_id', 'cat_id'])

    hier_summary = {}
    for level_name, res_df in hier_results.items():
        logger.info(f"\n--- Performance at {level_name} Level ---")
        print(res_df[['level', 'entity', 'sample_count', 'wape', 'rmse', 'mae', 'bias']].to_string(index=False))
        clean_records = json.loads(res_df.to_json(orient='records'))
        hier_summary[level_name] = clean_records

    with open(REPORTS_DIR / "hierarchical_performance.json", "w") as f:
        json.dump(hier_summary, f, indent=2)

    logger.info("=== STEP 3: Hierarchical Reconciliation Verification ===")
    # 1. Bottom-Up Reconciler
    bu_reconciler = BottomUpReconciler()
    reconciled_levels = bu_reconciler.reconcile(df, value_cols=['predicted', 'actual'])

    consistency = verify_hierarchy_consistency(reconciled_levels)
    logger.info(f"Bottom-Up Consistency Check Passed: {consistency['all_consistent']}")
    for k, v in consistency['checks'].items():
        logger.info(f"  Check '{k}': Consistent={v['consistent']}, Diff={v['diff']}")

    # 2. MinT Reconciler Verification
    logger.info("--- Testing MinT Reconciliation Engine ---")
    mint_reconciler = MinTReconciler(shrinkage_lambda=0.15)
    S, total_names, bottom_names = mint_reconciler.build_summing_matrix(df)
    logger.info(f"Summing Matrix S shape: {S.shape} (m={len(total_names)} total series, n_b={len(bottom_names)} bottom series)")

    # Aggregate bottom base forecasts to create full base vector
    y_bottom_sample = df.groupby('id')['predicted'].mean().loc[bottom_names].values
    y_base_full = S @ y_bottom_sample + np.random.normal(0, 0.05, len(total_names)) # Add small incoherent noise

    y_reconciled = mint_reconciler.reconcile(y_base_full)
    reconciled_bottom = y_reconciled[-len(bottom_names):]
    coherent_full = S @ reconciled_bottom

    mint_diff = np.max(np.abs(coherent_full - y_reconciled))
    logger.info(f"MinT Coherence Difference max(|S*y_b - y_all|): {mint_diff:.6e} (must be ~ 0)")

    recon_report = {
        "bottom_up_consistent": consistency['all_consistent'],
        "bottom_up_checks": consistency['checks'],
        "mint_summing_matrix_shape": list(S.shape),
        "mint_coherence_diff": float(mint_diff),
        "mint_verified": bool(mint_diff < 1e-4)
    }

    with open(REPORTS_DIR / "reconciliation_verification.json", "w") as f:
        json.dump(recon_report, f, indent=2)

    logger.info("Hierarchical evaluation and reconciliation verification finished successfully!")


if __name__ == "__main__":
    main()
