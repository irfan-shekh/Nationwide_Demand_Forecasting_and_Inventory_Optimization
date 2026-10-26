"""
Inventory Policy and Classical Operations Research Formulas
Calculates Safety Stock, Reorder Point (ROP), Lead-Time Demand,
and Expected Cost breakdowns under stochastic demand.
"""
from typing import Dict, Any, Optional
import numpy as np
from scipy.stats import norm


def calculate_safety_stock(
    demand_std: float,
    lead_time_days: int = 7,
    service_level: float = 0.95
) -> float:
    """
    Safety Stock formula for stochastic daily demand with deterministic lead time:
      SS = z * sigma_daily * sqrt(lead_time_days)
    where z = norm.ppf(service_level).
    """
    if demand_std <= 0 or lead_time_days <= 0:
        return 0.0
    z = norm.ppf(service_level)
    ss = z * demand_std * np.sqrt(lead_time_days)
    return float(max(0.0, np.ceil(ss)))


def calculate_reorder_point(
    mean_daily_demand: float,
    demand_std: float,
    lead_time_days: int = 7,
    service_level: float = 0.95
) -> Dict[str, float]:
    """
    Reorder Point (ROP) = Lead Time Demand + Safety Stock:
      ROP = (mean_daily_demand * lead_time_days) + SS
    """
    lt_demand = mean_daily_demand * lead_time_days
    ss = calculate_safety_stock(demand_std, lead_time_days, service_level)
    rop = lt_demand + ss

    return {
        "lead_time_demand": round(lt_demand, 2),
        "safety_stock": round(ss, 2),
        "reorder_point": round(rop, 2)
    }


def compute_expected_inventory_costs(
    order_quantity: float,
    current_stock: float,
    forecast_demand: float,
    unit_holding_cost: float,
    unit_stockout_penalty: float
) -> Dict[str, float]:
    """
    Computes ending inventory, expected shortages, and monetary holding/penalty costs.
    """
    available = current_stock + order_quantity
    net = available - forecast_demand

    ending_inventory = max(0.0, net)
    shortage = max(0.0, -net)

    holding_cost = ending_inventory * unit_holding_cost
    shortage_cost = shortage * unit_stockout_penalty
    total_cost = holding_cost + shortage_cost

    return {
        "ending_inventory": round(ending_inventory, 2),
        "expected_shortage": round(shortage, 2),
        "holding_cost": round(holding_cost, 2),
        "shortage_cost": round(shortage_cost, 2),
        "total_cost": round(total_cost, 2)
    }
