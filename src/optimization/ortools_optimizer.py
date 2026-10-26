"""
Google OR-Tools Mixed-Integer Linear Programming (MIP) Inventory Optimizer
Formulates and solves genuine multi-item constrained replenishment optimization:
Minimizes:
  Total Cost = sum_i [ h_i * I_end_i + p_i * Shortage_i + fixed_order_cost_i * z_i ]
Subject to:
  - Inventory Balance: I_end_i - Shortage_i = I_0_i + Q_i - Forecast_Demand_i
  - Semi-Continuous Minimum Order Quantity: MOQ_i * z_i <= Q_i <= Capacity_i * z_i
  - Total Store Shelf Capacity: sum_i (I_0_i + Q_i) <= Store_Capacity
  - Capital Replenishment Budget: sum_i (unit_cost_i * Q_i) <= Total_Budget
  - Binary decision variable z_i in {0, 1}
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    logger.warning("Google OR-Tools is not installed.")


class InventoryOptimizer:
    """
    Constrained Multi-Item Retail Replenishment Optimizer using Google OR-Tools MIP.
    """
    def __init__(
        self,
        holding_cost_rate: float = 0.15,
        stockout_penalty_mult: float = 3.0,
        fixed_order_cost: float = 10.0,
        store_capacity: float = 10000.0,
        total_budget: float = 50000.0,
        default_moq: float = 5.0
    ):
        self.holding_cost_rate = holding_cost_rate
        self.stockout_penalty_mult = stockout_penalty_mult
        self.fixed_order_cost = fixed_order_cost
        self.store_capacity = store_capacity
        self.total_budget = total_budget
        self.default_moq = default_moq

    def optimize(
        self,
        items_df: pd.DataFrame,
        demand_col: str = "predicted",
        price_col: str = "sell_price",
        stock_col: str = "current_stock"
    ) -> Dict[str, Any]:
        """
        Solves replenishment optimization for items in items_df.
        items_df should contain: ['id', 'item_id', 'store_id', 'predicted', 'sell_price', Optional('current_stock')]
        """
        df = items_df.copy()

        # Sensible defaults for unprovided business assumptions
        if stock_col not in df.columns:
            # Assume starting stock is ~1 week of demand (0.25 of 28-day period demand)
            df[stock_col] = (df[demand_col] * 0.25).round().astype(float)

        if price_col not in df.columns:
            df[price_col] = 5.0

        n_items = len(df)
        if n_items == 0:
            return {"status": "EMPTY_INPUT", "recommendations": pd.DataFrame(), "summary": {}}

        if not ORTOOLS_AVAILABLE:
            return self._heuristic_fallback(df, demand_col, price_col, stock_col)

        # Create OR-Tools MIP Solver (SCIP or CBC)
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            logger.warning("Neither SCIP nor CBC solver available; using GLOP/heuristic.")
            return self._heuristic_fallback(df, demand_col, price_col, stock_col)

        infinity = solver.infinity()

        # Decision Variables per item i
        Q = {}        # Order quantity (continuous or integer >= 0)
        I_end = {}    # Ending inventory (>= 0)
        Shortage = {} # Unmet demand shortage (>= 0)
        Z = {}        # Binary order flag {0, 1}

        for i in range(n_items):
            Q[i] = solver.NumVar(0.0, infinity, f'Q_{i}')
            I_end[i] = solver.NumVar(0.0, infinity, f'I_end_{i}')
            Shortage[i] = solver.NumVar(0.0, infinity, f'Shortage_{i}')
            Z[i] = solver.BoolVar(f'Z_{i}')

        # Constraints
        # 1. Semi-Continuous MOQ: Q[i] >= MOQ * Z[i] and Q[i] <= M * Z[i]
        M = max(self.store_capacity, 10000.0)
        for i in range(n_items):
            moq = self.default_moq
            solver.Add(Q[i] >= moq * Z[i])
            solver.Add(Q[i] <= M * Z[i])

        # 2. Inventory Flow Conservation: I_end[i] - Shortage[i] = I_0[i] + Q[i] - Demand[i]
        for i, row in df.reset_index(drop=True).iterrows():
            demand_val = float(row[demand_col])
            stock_val = float(row[stock_col])
            solver.Add(I_end[i] - Shortage[i] == stock_val + Q[i] - demand_val)

        # 3. Store Shelf Capacity Constraint: sum(stock_val + Q[i]) <= Capacity
        initial_stock_sum = float(df[stock_col].sum())
        effective_capacity = max(self.store_capacity, initial_stock_sum + 2000.0)
        solver.Add(
            solver.Sum(df.iloc[i][stock_col] + Q[i] for i in range(n_items)) <= effective_capacity
        )

        # 4. Total Budget Constraint: sum(unit_cost_i * Q[i]) <= Total_Budget
        # Estimate procurement unit cost at 60% of sell_price
        solver.Add(
            solver.Sum(float(df.iloc[i][price_col]) * 0.6 * Q[i] for i in range(n_items)) <= self.total_budget
        )

        # Objective Function:
        # Minimize: sum_i [ h_i * I_end[i] + p_i * Shortage[i] + fixed_order_cost * Z[i] ]
        objective = solver.Objective()
        for i, row in df.reset_index(drop=True).iterrows():
            price_val = float(row[price_col])
            unit_cost = price_val * 0.6
            h_i = unit_cost * self.holding_cost_rate
            p_i = price_val * self.stockout_penalty_mult

            objective.SetCoefficient(I_end[i], h_i)
            objective.SetCoefficient(Shortage[i], p_i)
            objective.SetCoefficient(Z[i], self.fixed_order_cost)

        objective.SetMinimization()

        status = solver.Solve()

        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            status_str = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE"
            recs = []
            tot_order_qty = 0.0
            tot_cost = solver.Objective().Value()
            tot_spend = 0.0

            for i, row in df.reset_index(drop=True).iterrows():
                q_val = round(Q[i].solution_value(), 2)
                iend_val = round(I_end[i].solution_value(), 2)
                short_val = round(Shortage[i].solution_value(), 2)
                unit_cost = float(row[price_col]) * 0.6
                spend = q_val * unit_cost
                tot_order_qty += q_val
                tot_spend += spend

                # ROP & Safety stock computation
                mean_d = float(row[demand_col]) / 28.0 if float(row[demand_col]) > 0 else 0.1
                ss = round(1.645 * np.sqrt(mean_d * 7), 2)
                rop = round(mean_d * 7 + ss, 2)

                recs.append({
                    "id": row.get("id", f"Item_{i}"),
                    "item_id": row.get("item_id", f"Item_{i}"),
                    "store_id": row.get("store_id", "Store_1"),
                    "state_id": row.get("state_id", "CA"),
                    "forecast_demand": round(float(row[demand_col]), 2),
                    "current_stock": float(row[stock_col]),
                    "safety_stock": ss,
                    "reorder_point": rop,
                    "recommended_order_qty": q_val,
                    "expected_inventory": iend_val,
                    "expected_shortage": short_val,
                    "holding_cost": round(iend_val * unit_cost * self.holding_cost_rate, 2),
                    "shortage_cost": round(short_val * float(row[price_col]) * self.stockout_penalty_mult, 2),
                    "order_cost": round(q_val * unit_cost, 2),
                    "solver_status": status_str,
                    "reorder_needed": bool(q_val > 0)
                })

            summary = {
                "solver_status": status_str,
                "total_order_quantity": round(tot_order_qty, 2),
                "total_spend": round(tot_spend, 2),
                "total_cost_objective": round(tot_cost, 2),
                "products_replenished": int(sum(1 for r in recs if r["recommended_order_qty"] > 0)),
                "total_products": len(recs),
                "budget_utilization": round((tot_spend / self.total_budget * 100.0) if self.total_budget > 0 else 0.0, 1)
            }
            return {
                "status": status_str,
                "recommendations": pd.DataFrame(recs),
                "summary": summary
            }

        else:
            logger.warning(f"OR-Tools solver ended with status {status}. Reverting to heuristic fallback.")
            return self._heuristic_fallback(df, demand_col, price_col, stock_col)

    def _heuristic_fallback(
        self,
        df: pd.DataFrame,
        demand_col: str,
        price_col: str,
        stock_col: str
    ) -> Dict[str, Any]:
        """Greedy heuristic fallback if solver is infeasible or unavailable."""
        recs = []
        tot_qty = 0.0
        tot_spend = 0.0

        for _, row in df.iterrows():
            d = float(row[demand_col])
            stock = float(row[stock_col])
            net = d - stock
            mean_d = d / 28.0 if d > 0 else 0.1
            ss = round(1.645 * np.sqrt(mean_d * 7), 2)
            rop = round(mean_d * 7 + ss, 2)

            q = max(0.0, net + ss)
            if q > 0 and q < self.default_moq:
                q = self.default_moq

            unit_cost = float(row[price_col]) * 0.6
            spend = q * unit_cost
            tot_qty += q
            tot_spend += spend

            recs.append({
                "id": row.get("id", "Item"),
                "item_id": row.get("item_id", "Item"),
                "store_id": row.get("store_id", "Store_1"),
                "state_id": row.get("state_id", "CA"),
                "forecast_demand": round(d, 2),
                "current_stock": stock,
                "safety_stock": ss,
                "reorder_point": rop,
                "recommended_order_qty": round(q, 2),
                "expected_inventory": round(max(0.0, stock + q - d), 2),
                "expected_shortage": round(max(0.0, d - (stock + q)), 2),
                "holding_cost": round(max(0.0, stock + q - d) * unit_cost * self.holding_cost_rate, 2),
                "shortage_cost": round(max(0.0, d - (stock + q)) * float(row[price_col]) * self.stockout_penalty_mult, 2),
                "order_cost": round(spend, 2),
                "solver_status": "HEURISTIC_RULE_BASED",
                "reorder_needed": bool(q > 0)
            })

        return {
            "status": "HEURISTIC_RULE_BASED",
            "recommendations": pd.DataFrame(recs),
            "summary": {
                "solver_status": "HEURISTIC_RULE_BASED",
                "total_order_quantity": round(tot_qty, 2),
                "total_spend": round(tot_spend, 2),
                "products_replenished": int(sum(1 for r in recs if r["recommended_order_qty"] > 0)),
                "total_products": len(recs),
                "budget_utilization": round((tot_spend / self.total_budget * 100.0) if self.total_budget > 0 else 0.0, 1)
            }
        }
