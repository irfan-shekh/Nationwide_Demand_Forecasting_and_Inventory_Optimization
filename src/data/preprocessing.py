"""
Data Preprocessing Pipeline: Wide-to-Long Transformation, Merging, and Cleaning
"""
from typing import Optional, List
import pandas as pd
import numpy as np
import logging
from src.data.loader import reduce_mem_usage
from src.data.validation import validate_sales_data, detect_intermittent_demand

logger = logging.getLogger(__name__)


def melt_sales_data(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms wide M5 sales dataframe (d_1, d_2, ...) into long format.
    Preserves all hierarchical IDs and downcasts integers.
    """
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    d_cols = [c for c in sales_df.columns if c.startswith('d_')]

    logger.info(f"Melting sales dataframe with {len(sales_df)} series and {len(d_cols)} days...")
    long_df = pd.melt(
        sales_df,
        id_vars=id_vars,
        value_vars=d_cols,
        var_name='d',
        value_name='sales'
    )
    long_df['sales'] = long_df['sales'].astype(np.int16)
    return reduce_mem_usage(long_df)


def merge_datasets(
    long_sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges long sales observations with calendar attributes and weekly sell prices.
    Uses exact key matches:
      - long_sales + calendar ON 'd'
      - resulting df + prices ON ['store_id', 'item_id', 'wm_yr_wk']
    """
    logger.info("Merging sales with calendar events and SNAP indicators...")
    cal_cols = [
        'date', 'wm_yr_wk', 'd', 'wday', 'month', 'year',
        'event_name_1', 'event_type_1', 'snap_CA', 'snap_TX', 'snap_WI'
    ]
    df = pd.merge(long_sales, calendar[cal_cols], on='d', how='left')

    logger.info("Merging with sell prices...")
    df = pd.merge(df, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    # Forward fill sell_price per item-store or fill with mean if newly introduced
    df['sell_price'] = df.groupby(['store_id', 'item_id'])['sell_price'].transform(
        lambda s: s.ffill().bfill()
    )
    # If still NaN (item never had listed price in that store), fill with item mean
    df['sell_price'] = df['sell_price'].fillna(df.groupby('item_id')['sell_price'].transform('mean'))
    df['sell_price'] = df['sell_price'].fillna(1.0).astype(np.float32)

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['id', 'date']).reset_index(drop=True)

    validate_sales_data(df)
    return reduce_mem_usage(df)


def analyze_intermittent_series(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes zero rate, ADI (Average Demand Interval), CV^2, and intermittency
    classification for each series.
    """
    records = []
    grouped = sales_df.groupby('id')
    for series_id, group in grouped:
        metrics = detect_intermittent_demand(group['sales'])
        item_id = group['item_id'].iloc[0]
        store_id = group['store_id'].iloc[0]
        dept_id = group['dept_id'].iloc[0]
        records.append({
            'id': series_id,
            'item_id': item_id,
            'store_id': store_id,
            'dept_id': dept_id,
            **metrics
        })
    return pd.DataFrame(records)
