"""
Centralized Configuration and Settings for Nationwide Demand Forecasting & Inventory Optimization
"""
from pathlib import Path
import os

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
SAMPLE_DATA_DIR = BASE_DIR / "data" / "sample"

def _resolve_data_path(filename: str) -> Path:
    raw_p = RAW_DATA_DIR / filename
    if raw_p.exists():
        return raw_p
    return BASE_DIR / filename

# Raw M5 Files (prioritizes data/raw/ then project root)
CALENDAR_PATH = _resolve_data_path("calendar.csv")
SALES_TRAIN_VAL_PATH = _resolve_data_path("sales_train_validation.csv")
SALES_TRAIN_EVAL_PATH = _resolve_data_path("sales_train_evaluation.csv")
PRICES_PATH = _resolve_data_path("sell_prices.csv")
SAMPLE_SUBMISSION_PATH = _resolve_data_path("sample_submission.csv")

# Model & Artifact Directories
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
TFT_ARTIFACTS_DIR = BASE_DIR / "artifacts" / "tft"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"

# Forecasting Configuration
FORECAST_HORIZON = 28  # 28 days standard M5 horizon
VALIDATION_DAYS = 28   # Last 28 days of validation set for backtesting
RANDOM_SEED = 42

# Default Stratified Sample Settings for Local Dev / Dashboard Speed
# 50 top items across all 10 stores (or 10 representative items x 10 stores = 100 series)
SAMPLE_NUM_ITEMS = 30
SAMPLE_STORES = ["CA_1", "CA_2", "TX_1", "WI_1"]

# Inventory Optimization Defaults (Sensible Retail Business Assumptions)
DEFAULT_LEAD_TIME = 7          # Days
DEFAULT_SERVICE_LEVEL = 0.95   # 95% cycle service level (Z = 1.645)
DEFAULT_HOLDING_COST_RATE = 0.15 # 15% annual or $0.15/unit/week
DEFAULT_STOCKOUT_PENALTY_MULT = 3.0 # Stockout penalty is 3x unit margin
DEFAULT_MIN_ORDER_QTY = 5      # Units
DEFAULT_CAPACITY_LIMIT = 5000  # Total store units
DEFAULT_BUDGET_LIMIT = 25000.0 # Total capital replenishment budget in USD

# Column Names & Hierarchy
ID_COL = "id"
ITEM_COL = "item_id"
DEPT_COL = "dept_id"
CAT_COL = "cat_id"
STORE_COL = "store_id"
STATE_COL = "state_id"
DATE_COL = "date"
DEMAND_COL = "sales"

HIERARCHY_LEVELS = [
    "national",
    "state_id",
    "store_id",
    "dept_id",
    "cat_id",
    "item_id"
]
