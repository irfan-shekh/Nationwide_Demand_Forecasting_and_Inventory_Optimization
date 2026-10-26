"""
Master Pipeline Runner Script
Orchestrates end-to-end execution:
1. Data validation and preparation (prepare_data.py)
2. Forecasting model training and benchmarking (train_models.py)
3. Hierarchical evaluation and MinT reconciliation (evaluate_models.py)
4. Google OR-Tools replenishment inventory optimization test
"""
import sys
from pathlib import Path
import json
import logging
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.prepare_data import main as run_prepare
from scripts.train_models import main as run_train
from scripts.evaluate_models import main as run_evaluate
from src.optimization.ortools_optimizer import InventoryOptimizer
from config.settings import SAMPLE_DATA_DIR, REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("===============================================================")
    logger.info("STARTING END-TO-END PIPELINE: NATIONWIDE DEMAND FORECASTING")
    logger.info("===============================================================")

    logger.info("\n>>> PHASE 1: Data Preparation & Quality Reporting <<<")
    run_prepare()

    logger.info("\n>>> PHASE 2: Model Training & Holdout Benchmarking <<<")
    run_train()

    logger.info("\n>>> PHASE 3: Hierarchical Evaluation & MinT Reconciliation <<<")
    run_evaluate()

    logger.info("\n>>> PHASE 4: Google OR-Tools Inventory Optimization <<<")
    pred_path = SAMPLE_DATA_DIR / "test_predictions.parquet"
    pred_df = pd.read_parquet(pred_path)

    # Aggregate 28-day forecast demand per series
    item_demand = pred_df.groupby(['id', 'item_id', 'store_id', 'state_id']).agg({
        'pred_lightgbm': 'sum',
        'sell_price': 'mean'
    }).reset_index().rename(columns={'pred_lightgbm': 'predicted'})

    optimizer = InventoryOptimizer(
        holding_cost_rate=0.15,
        stockout_penalty_mult=3.0,
        fixed_order_cost=10.0,
        store_capacity=5000.0,
        total_budget=25000.0,
        default_moq=5.0
    )
    opt_results = optimizer.optimize(item_demand)
    logger.info(f"OR-Tools Solver Status: {opt_results['status']}")
    logger.info(f"Summary: {json.dumps(opt_results['summary'], indent=2)}")

    recs_path = SAMPLE_DATA_DIR / "inventory_recommendations.parquet"
    opt_results['recommendations'].to_parquet(recs_path, index=False)

    with open(REPORTS_DIR / "inventory_optimization_summary.json", "w") as f:
        json.dump(opt_results['summary'], f, indent=2)

    logger.info("===============================================================")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("===============================================================")


if __name__ == "__main__":
    main()
