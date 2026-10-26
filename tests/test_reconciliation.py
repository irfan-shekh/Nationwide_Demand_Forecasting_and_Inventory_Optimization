"""
Unit tests for hierarchical reconciliation (Bottom-Up and MinT).
"""
import pytest
import pandas as pd
import numpy as np
from src.forecasting.reconciliation import BottomUpReconciler, MinTReconciler, verify_hierarchy_consistency


def test_bottom_up_reconciliation_exact_sum():
    # 2 items in 2 stores in 2 states
    data = pd.DataFrame({
        "id": ["ITEM1_CA1", "ITEM2_CA1", "ITEM1_TX1", "ITEM2_TX1"],
        "item_id": ["ITEM1", "ITEM2", "ITEM1", "ITEM2"],
        "store_id": ["CA_1", "CA_1", "TX_1", "TX_1"],
        "state_id": ["CA", "CA", "TX", "TX"],
        "dept_id": ["FOODS_1", "FOODS_1", "FOODS_1", "FOODS_1"],
        "cat_id": ["FOODS", "FOODS", "FOODS", "FOODS"],
        "date": pd.to_datetime(["2026-01-01"] * 4),
        "predicted": [10.0, 15.0, 20.0, 25.0],
        "sales": [9.0, 14.0, 21.0, 26.0]
    })

    reconciler = BottomUpReconciler()
    levels = reconciler.reconcile(data)

    # State level checks
    ca_pred = levels["State"].set_index("state_id").loc["CA", "predicted"]
    tx_pred = levels["State"].set_index("state_id").loc["TX", "predicted"]
    assert ca_pred == 25.0  # 10 + 15
    assert tx_pred == 45.0  # 20 + 25

    # National level check
    nat_pred = levels["National"]["predicted"].iloc[0]
    assert nat_pred == 70.0  # 25 + 45

    # Consistency check
    consistency = verify_hierarchy_consistency(levels)
    assert consistency["all_consistent"] is True


def test_mint_summing_matrix_and_coherence():
    bottom_df = pd.DataFrame({
        "id": ["SERIES_1", "SERIES_2", "SERIES_3"],
        "store_id": ["CA_1", "CA_1", "TX_1"],
        "state_id": ["CA", "CA", "TX"]
    })

    mint = MinTReconciler(shrinkage_lambda=0.1)
    S, total_names, bottom_names = mint.build_summing_matrix(bottom_df)

    # Bottom series: 3. Hierarchy rows: National (1) + States (2: CA, TX) + Stores (2: CA_1, TX_1) + Bottom (3) = 8 rows
    assert S.shape == (8, 3)

    # National row must sum all bottom items
    assert np.all(S[0, :] == 1.0)

    # Test projection with arbitrary base forecasts
    y_base = np.array([100.0, 60.0, 40.0, 60.0, 40.0, 25.0, 35.0, 40.0])
    y_reconciled = mint.reconcile(y_base)

    # Reconciled bottom series projected via S must exactly equal reconciled full hierarchy
    y_bottom_reconciled = y_reconciled[-3:]
    projected_full = S @ y_bottom_reconciled
    np.testing.assert_allclose(projected_full, y_reconciled, atol=1e-5)
