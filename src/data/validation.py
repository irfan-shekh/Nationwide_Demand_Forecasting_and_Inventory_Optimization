"""
Data Validation and Quality Reporting Module
Validates schemas, data types, missing values, duplicates, and range anomalies.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityReport:
    """Holds data validation findings and summary statistics."""
    def __init__(self, name: str):
        self.name = name
        self.row_count: int = 0
        self.col_count: int = 0
        self.duplicate_count: int = 0
        self.missing_summary: Dict[str, int] = {}
        self.data_types: Dict[str, str] = {}
        self.warnings: List[str] = []
        self.passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.name,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "duplicate_count": self.duplicate_count,
            "missing_summary": self.missing_summary,
            "data_types": self.data_types,
            "warnings": self.warnings,
            "passed": self.passed,
        }


def validate_schema(df: pd.DataFrame, expected_cols: List[str], dataset_name: str = "Dataset") -> DataQualityReport:
    """
    Validates that a DataFrame contains expected columns and reports on data quality.
    """
    report = DataQualityReport(dataset_name)
    report.row_count = len(df)
    report.col_count = len(df.columns)
    report.duplicate_count = int(df.duplicated().sum())
    report.data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Check required columns
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        report.warnings.append(f"Missing required columns: {missing_cols}")
        report.passed = False

    # Check nulls
    null_counts = df.isnull().sum()
    report.missing_summary = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}

    if report.duplicate_count > 0:
        report.warnings.append(f"Found {report.duplicate_count} duplicate rows.")

    logger.info(f"[{dataset_name}] Validation completed. Passed: {report.passed}, Rows: {report.row_count}, Warnings: {len(report.warnings)}")
    return report


def validate_sales_data(df: pd.DataFrame) -> DataQualityReport:
    """Validates unpivoted or long sales time-series data."""
    report = validate_schema(df, ["date", "item_id", "store_id", "sales"], "SalesTimeSeries")

    # Sales non-negativity check
    if "sales" in df.columns:
        negative_sales = (df["sales"] < 0).sum()
        if negative_sales > 0:
            report.warnings.append(f"Found {negative_sales} negative sales values. Sales must be non-negative.")
            report.passed = False

    # Date continuity check per series
    if "date" in df.columns and "id" in df.columns:
        dup_series_date = df.duplicated(subset=["id", "date"]).sum()
        if dup_series_date > 0:
            report.warnings.append(f"Found {dup_series_date} duplicate series-date combinations.")
            report.passed = False

    return report


def detect_intermittent_demand(series: pd.Series) -> Dict[str, float]:
    """
    Calculates zero-demand frequency and average demand size when non-zero.
    Useful for retail demand categorization (Syntetos-Boylan / Croston classification).
    """
    if len(series) == 0:
        return {"zero_rate": 0.0, "mean_non_zero": 0.0, "adi": 0.0, "cv2": 0.0}

    zero_count = (series == 0).sum()
    zero_rate = float(zero_count / len(series))
    non_zeros = series[series > 0]

    mean_non_zero = float(non_zeros.mean()) if len(non_zeros) > 0 else 0.0
    # Average Demand Interval (ADI) = N / count(non_zeros)
    adi = float(len(series) / len(non_zeros)) if len(non_zeros) > 0 else float("inf")
    # Squared Coefficient of Variation (CV^2)
    cv2 = float((non_zeros.std() / mean_non_zero) ** 2) if (len(non_zeros) > 1 and mean_non_zero > 0) else 0.0

    return {
        "zero_rate": round(zero_rate, 4),
        "mean_non_zero": round(mean_non_zero, 4),
        "adi": round(adi, 4),
        "cv2": round(cv2, 4),
        "is_intermittent": bool(adi > 1.32 or zero_rate > 0.4)
    }
