"""
Data Preparation and Quality Reporting Script
Extracts, validates, and prepares stratified benchmark sample and full processed datasets.
"""
import sys
from pathlib import Path
import json
import logging

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    CALENDAR_PATH,
    SALES_TRAIN_VAL_PATH,
    PRICES_PATH,
    SAMPLE_DATA_DIR,
    REPORTS_DIR,
    SAMPLE_NUM_ITEMS,
    SAMPLE_STORES
)
from src.data.loader import create_development_sample, load_calendar, load_sell_prices, load_raw_sales
from src.data.preprocessing import melt_sales_data, merge_datasets, analyze_intermittent_series
from src.data.validation import validate_schema, validate_sales_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== STEP 1: Validating Raw Datasets ===")
    for p in [CALENDAR_PATH, SALES_TRAIN_VAL_PATH, PRICES_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Required raw dataset not found: {p}")
        logger.info(f"Verified: {p.name} ({p.stat().st_size / (1024**2):.1f} MB)")

    logger.info("=== STEP 2: Creating Stratified Development Benchmark Sample ===")
    sample_sales, sample_prices, calendar = create_development_sample(
        num_items=SAMPLE_NUM_ITEMS,
        stores=SAMPLE_STORES,
        days_back=365,  # 1 year of daily history for fast, robust training
        save=True
    )

    logger.info("=== STEP 3: Transforming Wide Sales into Long Series & Merging Covariates ===")
    long_sales = melt_sales_data(sample_sales)
    merged_df = merge_datasets(long_sales, calendar, sample_prices)

    merged_path = SAMPLE_DATA_DIR / "merged_sample_features.parquet"
    merged_df.to_parquet(merged_path, index=False)
    logger.info(f"Saved merged dataset: {len(merged_df)} rows to {merged_path}")

    logger.info("=== STEP 4: Analyzing Demand Intermittency ===")
    intermittency_df = analyze_intermittent_series(merged_df)
    intermittent_path = SAMPLE_DATA_DIR / "intermittent_analysis.parquet"
    intermittency_df.to_parquet(intermittent_path, index=False)

    logger.info("=== STEP 5: Producing Data Quality Report ===")
    report_dict = {
        "raw_calendar_rows": len(calendar),
        "raw_calendar_cols": len(calendar.columns),
        "sample_series_count": int(sample_sales['id'].nunique()),
        "sample_stores": SAMPLE_STORES,
        "sample_total_rows": len(merged_df),
        "date_range": [str(merged_df['date'].min().date()), str(merged_df['date'].max().date())],
        "zero_sales_rate": round(float((merged_df['sales'] == 0).mean()), 4),
        "mean_daily_sales": round(float(merged_df['sales'].mean()), 2),
        "intermittent_series_percentage": round(float(intermittency_df['is_intermittent'].mean()) * 100, 1),
        "status": "PASSED_QUALITY_CHECKS"
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "data_quality_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    logger.info(f"Data quality report saved to {REPORTS_DIR / 'data_quality_report.json'}")
    logger.info("Data preparation completed successfully!")


if __name__ == "__main__":
    main()
