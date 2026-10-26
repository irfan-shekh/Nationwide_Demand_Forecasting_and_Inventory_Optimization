"""
Unit tests for Inventory Policy calculations and Google OR-Tools Optimizer.
"""
import pytest
import pandas as pd
import numpy as np
from src.optimization.inventory_policy import (
    calculate_safety_stock,
    calculate_reorder_point,
    compute_expected_inventory_costs
)
from src.optimization.ortools_optimizer import InventoryOptimizer, ORTOOLS_AVAILABLE


def test_safety_stock_and_reorder_point():
    # Mean daily demand = 10, std = 3, lead time = 4 days, 95% service level (z = 1.645)
    # SS = ceil(1.645 * 3 * sqrt(4)) = ceil(1.645 * 3 * 2) = ceil(9.87) = 10.0
    ss = calculate_safety_stock(demand_std=3.0, lead_time_days=4, service_level=0.95)
    assert ss == 10.0

    rop_dict = calculate_reorder_point(mean_daily_demand=10.0, demand_std=3.0, lead_time_days=4, service_level=0.95)
    assert rop_dict["lead_time_demand"] == 40.0
    assert rop_dict["safety_stock"] == 10.0
    assert rop_dict["reorder_point"] == 50.0


def test_inventory_cost_breakdown():
    # Order 20, stock 10 -> total 30. Demand 25 -> Ending = 5, Shortage = 0
    res = compute_expected_inventory_costs(
        order_quantity=20.0,
        current_stock=10.0,
        forecast_demand=25.0,
        unit_holding_cost=1.0,
        unit_stockout_penalty=5.0
    )
    assert res["ending_inventory"] == 5.0
    assert res["expected_shortage"] == 0.0
    assert res["holding_cost"] == 5.0
    assert res["shortage_cost"] == 0.0


def test_ortools_optimization_constraints():
    # 3 items with demands
    items_df = pd.DataFrame({
        "id": ["ITEM_1", "ITEM_2", "ITEM_3"],
        "item_id": ["ITEM_1", "ITEM_2", "ITEM_3"],
        "store_id": ["CA_1", "CA_1", "CA_1"],
        "predicted": [50.0, 100.0, 30.0],
        "sell_price": [10.0, 20.0, 5.0],
        "current_stock": [10.0, 20.0, 5.0]
    })

    optimizer = InventoryOptimizer(
        holding_cost_rate=0.15,
        stockout_penalty_mult=3.0,
        store_capacity=500.0,
        total_budget=5000.0,
        default_moq=5.0
    )
    res = optimizer.optimize(items_df)

    assert res["status"] in ["OPTIMAL", "FEASIBLE"]
    recs = res["recommendations"]

    # Constraint 1: Non-negativity
    assert (recs["recommended_order_qty"] >= 0).all()

    # Constraint 2: Semi-continuous MOQ (if order > 0, order >= 5.0)
    for q in recs["recommended_order_qty"]:
        if q > 0:
            assert q >= 5.0 - 1e-4

    # Constraint 3: Capacity
    total_inventory = (recs["current_stock"] + recs["recommended_order_qty"]).sum()
    assert total_inventory <= 500.0 + 1e-4

    # Constraint 4: Budget
    assert res["summary"]["total_spend"] <= 5000.0 + 1e-4


def test_ortools_infeasible_budget_fallback():
    # Budget set to $0.00
    items_df = pd.DataFrame({
        "id": ["ITEM_1"],
        "item_id": ["ITEM_1"],
        "store_id": ["CA_1"],
        "predicted": [50.0],
        "sell_price": [10.0],
        "current_stock": [0.0]
    })

    optimizer = InventoryOptimizer(
        store_capacity=100.0,
        total_budget=0.0, # Zero budget
        default_moq=10.0
    )
    res = optimizer.optimize(items_df)
    # Should solve with 0 order quantity without crashing
    assert res["status"] in ["OPTIMAL", "FEASIBLE", "HEURISTIC_RULE_BASED"]
    assert res["summary"]["total_spend"] <= 1e-4
