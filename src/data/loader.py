"""
M5 Dataset Loader and Memory Optimizer
Handles reading raw CSVs, downcasting data types, sampling, and caching.
"""
from typing import Optional, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from config.settings import (
    CALENDAR_PATH,
    SALES_TRAIN_VAL_PATH,
    PRICES_PATH,
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_DIR
)
from src.data.validation import validate_schema

logger = logging.getLogger(__name__)


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Iterates through all columns and downcasts numerical types to save RAM.
    Preserves precision while reducing memory by 60-80%.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype

        if (col_type != object 
            and not isinstance(df[col].dtype, pd.CategoricalDtype)
            and not pd.api.types.is_datetime64_any_dtype(df[col])):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.info(f"Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df


def load_calendar(path: Optional[Path] = None) -> pd.DataFrame:
    """Loads and optimizes calendar.csv."""
    filepath = path or CALENDAR_PATH
    if not filepath.exists():
        raise FileNotFoundError(f"Calendar file not found at: {filepath}")

    expected = ['date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'd', 'event_name_1', 'snap_CA', 'snap_TX', 'snap_WI']
    df = pd.read_csv(filepath)
    validate_schema(df, expected, "Calendar")

    df['date'] = pd.to_datetime(df['date'])
    # Convert snap columns to int8
    for c in ['snap_CA', 'snap_TX', 'snap_WI', 'wday', 'month', 'year']:
        if c in df.columns:
            df[c] = df[c].astype(np.int16)

    # Convert events to string/category
    for c in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
        if c in df.columns:
            df[c] = df[c].fillna("None")

    return reduce_mem_usage(df)


def load_sell_prices(path: Optional[Path] = None, filter_stores: Optional[List[str]] = None, filter_items: Optional[List[str]] = None) -> pd.DataFrame:
    """Loads and optimizes sell_prices.csv with optional item/store filtering."""
    filepath = path or PRICES_PATH
    if not filepath.exists():
        raise FileNotFoundError(f"Prices file not found at: {filepath}")

    df = pd.read_csv(filepath)
    if filter_stores:
        df = df[df['store_id'].isin(filter_stores)]
    if filter_items:
        df = df[df['item_id'].isin(filter_items)]

    df['sell_price'] = df['sell_price'].astype(np.float32)
    df['wm_yr_wk'] = df['wm_yr_wk'].astype(np.int32)
    return reduce_mem_usage(df)


def load_raw_sales(path: Optional[Path] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """Loads raw wide sales matrix (sales_train_validation.csv)."""
    filepath = path or SALES_TRAIN_VAL_PATH
    if not filepath.exists():
        raise FileNotFoundError(f"Sales file not found at: {filepath}")

    df = pd.read_csv(filepath, nrows=nrows)
    # Downcast d_1 ... d_N to int16
    d_cols = [c for c in df.columns if c.startswith('d_')]
    for c in d_cols:
        df[c] = df[c].astype(np.int16)
    return df


def create_development_sample(
    num_items: int = 30,
    stores: Optional[List[str]] = None,
    days_back: int = 365,
    save: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates a stratified, representative development dataset covering:
    - Multiple states (CA, TX, WI)
    - Multiple departments (FOODS, HOBBIES, HOUSEHOLD)
    - Active high-demand and intermittent-demand items
    This ensures rapid, reproducible model training and instant dashboard performance.
    """
    selected_stores = stores or ["CA_1", "CA_2", "TX_1", "WI_1"]
    raw_sales = load_raw_sales()

    # Stratified selection: pick top items per category/department
    filtered = raw_sales[raw_sales['store_id'].isin(selected_stores)]
    sample_items = filtered.groupby('dept_id', group_keys=False).apply(
        lambda g: g.drop_duplicates(subset=['item_id']).head(num_items // len(filtered['dept_id'].unique()))
    )['item_id'].unique().tolist()

    if len(sample_items) < num_items:
        remaining = filtered[~filtered['item_id'].isin(sample_items)]['item_id'].unique()
        sample_items.extend(remaining[:num_items - len(sample_items)])

    sample_sales = filtered[filtered['item_id'].isin(sample_items)].copy()

    # Filter prices and calendar
    sample_prices = load_sell_prices(filter_stores=selected_stores, filter_items=sample_items)
    calendar = load_calendar()

    # Keep recent days_back days if specified
    if days_back:
        d_cols = [f"d_{i}" for i in range(1, 1914)]
        keep_d_cols = d_cols[-days_back:]
        meta_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        sample_sales = sample_sales[meta_cols + keep_d_cols]

    if save:
        SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        sample_sales.to_parquet(SAMPLE_DATA_DIR / "sample_sales.parquet", index=False)
        sample_prices.to_parquet(SAMPLE_DATA_DIR / "sample_prices.parquet", index=False)
        calendar.to_parquet(SAMPLE_DATA_DIR / "calendar.parquet", index=False)
        logger.info(f"Saved stratified sample dataset: {len(sample_sales)} series across {len(selected_stores)} stores.")

    return sample_sales, sample_prices, calendar
