"""
Dashboard Reactive Callbacks Implementation
Connects UI filter controls, charts, tables, and live OR-Tools re-optimization.
"""
from dash import Input, Output, State, html, dash_table
import pandas as pd
import numpy as np
from pathlib import Path
import json

from config.settings import SAMPLE_DATA_DIR, REPORTS_DIR
from dashboard.charts import (
    create_demand_forecast_chart,
    create_model_comparison_bar,
    create_state_comparison_chart,
    create_top_products_chart,
    create_inventory_cost_breakdown
)
from src.optimization.ortools_optimizer import InventoryOptimizer


def register_callbacks(app, merged_df: pd.DataFrame, pred_df: pd.DataFrame):
    """Registers all reactive dash callbacks on the app instance."""

    # 1. Main Tab Switching Callback
    @app.callback(
        Output("tab-content-container", "children"),
        Input("main-tabs", "active_tab")
    )
    def switch_tab(tab_id):
        from dashboard.layout import (
            render_overview_tab,
            render_forecasting_tab,
            render_comparison_tab,
            render_inventory_tab,
            render_hierarchy_tab,
            render_explorer_tab,
            render_status_tab
        )
        stores = sorted(merged_df['store_id'].unique().tolist())
        depts = sorted(merged_df['dept_id'].unique().tolist())
        items = sorted(merged_df['item_id'].unique().tolist())

        if tab_id == "tab-forecasting":
            return render_forecasting_tab(stores, depts, items)
        elif tab_id == "tab-comparison":
            return render_comparison_tab()
        elif tab_id == "tab-inventory":
            return render_inventory_tab()
        elif tab_id == "tab-hierarchy":
            return render_hierarchy_tab()
        elif tab_id == "tab-explorer":
            return render_explorer_tab(stores, depts)
        elif tab_id == "tab-status":
            return render_status_tab({"mode": "MODE_2_EXPORTED_FORECASTS"})
        return render_overview_tab()

    # 2. Executive Overview Tab Charts
    @app.callback(
        [Output("overview-trend-chart", "figure"),
         Output("overview-state-chart", "figure"),
         Output("overview-top-items-chart", "figure")],
        Input("main-tabs", "active_tab")
    )
    def render_overview_charts(active_tab):
        if active_tab != "tab-overview":
            from dash.exceptions import PreventUpdate
            raise PreventUpdate

        # Daily National Aggregation
        nat_hist = merged_df.groupby('date')['sales'].sum().reset_index()
        nat_fc = pred_df.groupby('date').agg({
            'actual': 'sum',
            'pred_lightgbm': 'sum',
            'pred_lightgbm_lower': 'sum',
            'pred_lightgbm_upper': 'sum'
        }).reset_index().rename(columns={
            'pred_lightgbm': 'predicted',
            'pred_lightgbm_lower': 'pred_lower',
            'pred_lightgbm_upper': 'pred_upper'
        })

        trend_fig = create_demand_forecast_chart(
            nat_hist,
            nat_fc,
            title="National Aggregated Demand: History & 28-Day Forecast"
        )
        state_fig = create_state_comparison_chart(merged_df)
        top_fig = create_top_products_chart(merged_df, top_n=10)

        return trend_fig, state_fig, top_fig

    # 3. Demand Forecasting Tab Interactive Chart
    @app.callback(
        [Output("forecast-series-chart", "figure"),
         Output("fc-provenance-badge", "children"),
         Output("fc-metrics-summary", "children")],
        [Input("fc-store-filter", "value"),
         Input("fc-item-filter", "value"),
         Input("fc-model-filter", "value"),
         Input("fc-horizon-filter", "value")]
    )
    def update_forecast_chart(store_id, item_id, model_name, horizon):
        if not store_id or not item_id:
            from dash.exceptions import PreventUpdate
            raise PreventUpdate

        series_id = f"{item_id}_{store_id}_validation"
        hist_series = merged_df[(merged_df['store_id'] == store_id) & (merged_df['item_id'] == item_id)].sort_values('date')
        fc_series = pred_df[(pred_df['store_id'] == store_id) & (pred_df['item_id'] == item_id)].sort_values('date').head(horizon or 28).copy()

        if fc_series.empty:
            # Fallback filter on item alone
            hist_series = merged_df[merged_df['item_id'] == item_id].groupby('date')['sales'].sum().reset_index()
            fc_series = pred_df[pred_df['item_id'] == item_id].groupby('date').agg({
                'actual': 'sum',
                'pred_lightgbm': 'sum',
                'pred_lightgbm_lower': 'sum',
                'pred_lightgbm_upper': 'sum',
                'pred_tft': 'sum',
                'pred_snaive': 'sum',
                'pred_naive': 'sum'
            }).reset_index().head(horizon or 28)

        # Select model predictions
        if model_name == "LightGBM":
            fc_series['predicted'] = fc_series['pred_lightgbm']
            fc_series['pred_lower'] = fc_series.get('pred_lightgbm_lower', fc_series['predicted'] * 0.75)
            fc_series['pred_upper'] = fc_series.get('pred_lightgbm_upper', fc_series['predicted'] * 1.25)
            badge_text = "LightGBM Gradient Booster (models/lightgbm_model.pkl)"
        elif model_name == "TFT":
            fc_series['predicted'] = fc_series.get('pred_tft', fc_series['pred_lightgbm'])
            fc_series['pred_lower'] = fc_series.get('pred_tft_lower', fc_series['predicted'] * 0.7)
            fc_series['pred_upper'] = fc_series.get('pred_tft_upper', fc_series['predicted'] * 1.3)
            badge_text = "Temporal Fusion Transformer (artifacts/tft/ - Colab Mode 2)"
        elif model_name == "Seasonal-Naive":
            fc_series['predicted'] = fc_series.get('pred_snaive', fc_series['actual'])
            fc_series['pred_lower'] = np.clip(fc_series['predicted'] * 0.65, 0, None)
            fc_series['pred_upper'] = fc_series['predicted'] * 1.35
            badge_text = "Seasonal-Naive 7-Day Cycle Benchmark"
        else:
            fc_series['predicted'] = fc_series.get('pred_naive', fc_series['actual'])
            fc_series['pred_lower'] = np.clip(fc_series['predicted'] * 0.7, 0, None)
            fc_series['pred_upper'] = fc_series['predicted'] * 1.3
            badge_text = "Naive Last-Value Persistence Benchmark"

        fig = create_demand_forecast_chart(
            hist_series.tail(60), # show last 60 days history
            fc_series,
            title=f"Forecast Demand for {item_id} at Store {store_id} ({horizon or 28}-Day Horizon)"
        )

        # Compute series-specific metrics
        from src.evaluation.metrics import compute_all_metrics
        m = compute_all_metrics(fc_series['actual'], fc_series['predicted'])
        metrics_p = html.P([
            html.Strong("Series Holdout Metrics: "),
            f"WAPE: {m['wape']:.4f}  |  RMSE: {m['rmse']:.4f}  |  MAE: {m['mae']:.4f}  |  Signed Bias: {m['bias']:.4f} units"
        ], className="text-muted small")

        return fig, badge_text, metrics_p

    # 4. Model Comparison Tab
    @app.callback(
        [Output("comparison-table-container", "children"),
         Output("comparison-wape-chart", "figure"),
         Output("comparison-rmse-chart", "figure")],
        Input("main-tabs", "active_tab")
    )
    def render_comparison_view(active_tab):
        if active_tab != "tab-comparison":
            from dash.exceptions import PreventUpdate
            raise PreventUpdate

        comp_path = REPORTS_DIR / "model_comparison.csv"
        if comp_path.exists():
            comp_df = pd.read_csv(comp_path)
        else:
            comp_df = pd.DataFrame()

        table = dash_table.DataTable(
            data=comp_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in comp_df.columns],
            style_header={'backgroundColor': '#0F172A', 'color': 'white', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_data_conditional=[
                {'if': {'row_index': 0}, 'backgroundColor': '#EFF6FF', 'fontWeight': 'bold'}
            ]
        )
        wape_chart = create_model_comparison_bar(comp_df, metric='wape')
        rmse_chart = create_model_comparison_bar(comp_df, metric='rmse')

        return table, wape_chart, rmse_chart

    # 5. Interactive OR-Tools Re-Optimization Callback
    @app.callback(
        [Output("inventory-recommendation-table", "children"),
         Output("inventory-cost-chart", "figure")],
        [Input("inv-reoptimize-btn", "n_clicks")],
        [State("inv-lead-time-slider", "value"),
         State("inv-service-level-slider", "value"),
         State("inv-holding-rate-slider", "value"),
         State("inv-penalty-slider", "value"),
         State("inv-capacity-input", "value"),
         State("inv-budget-input", "value")]
    )
    def run_live_reoptimization(n_clicks, lead_time, service_level, holding_rate, penalty_mult, capacity, budget):
        # Aggregate demand per series
        item_demand = pred_df.groupby(['id', 'item_id', 'store_id', 'state_id']).agg({
            'pred_lightgbm': 'sum',
            'sell_price': 'mean'
        }).reset_index().rename(columns={'pred_lightgbm': 'predicted'})

        optimizer = InventoryOptimizer(
            holding_cost_rate=(holding_rate or 15) / 100.0,
            stockout_penalty_mult=float(penalty_mult or 3.0),
            store_capacity=float(capacity or 5000),
            total_budget=float(budget or 25000),
            default_moq=5.0
        )
        res = optimizer.optimize(item_demand)
        recs_df = res['recommendations']

        cost_fig = create_inventory_cost_breakdown(recs_df)

        display_cols = ['item_id', 'store_id', 'forecast_demand', 'safety_stock', 'reorder_point', 'recommended_order_qty', 'expected_inventory', 'expected_shortage', 'holding_cost', 'shortage_cost', 'solver_status']
        avail_cols = [c for c in display_cols if c in recs_df.columns]

        table = dash_table.DataTable(
            data=recs_df[avail_cols].head(30).to_dict('records'),
            columns=[{"name": c.replace('_', ' ').title(), "id": c} for c in avail_cols],
            page_size=10,
            style_header={'backgroundColor': '#0F172A', 'color': 'white', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'center', 'padding': '8px'},
            style_data_conditional=[
                {'if': {'filter_query': '{recommended_order_qty} > 0'}, 'backgroundColor': '#FEF3C7', 'color': '#92400E'}
            ]
        )
        return table, cost_fig

    # 6. Hierarchical Metrics Table Callback
    @app.callback(
        Output("hierarchy-metrics-table", "children"),
        Input("main-tabs", "active_tab")
    )
    def render_hierarchy_table(active_tab):
        if active_tab != "tab-hierarchy":
            from dash.exceptions import PreventUpdate
            raise PreventUpdate

        hier_file = REPORTS_DIR / "hierarchical_performance.json"
        if hier_file.exists():
            with open(hier_file, "r") as f:
                data = json.load(f)
            flat = []
            for k, records in data.items():
                flat.extend(records)
            df = pd.DataFrame(flat)
            return dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{"name": c.title(), "id": c} for c in df.columns],
                page_size=10,
                style_header={'backgroundColor': '#0F172A', 'color': 'white', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'center', 'padding': '8px'}
            )
        return html.P("No hierarchical report found. Run evaluate_models.py.")

    # 7. Data Explorer Table Callback
    @app.callback(
        Output("explorer-table-container", "children"),
        [Input("exp-store-filter", "value"),
         Input("exp-dept-filter", "value"),
         Input("exp-pagesize-filter", "value")]
    )
    def update_explorer_table(store, dept, page_size):
        df = merged_df.copy()
        if store and store != "ALL":
            df = df[df['store_id'] == store]
        if dept and dept != "ALL":
            df = df[df['dept_id'] == dept]

        view_cols = ['date', 'state_id', 'store_id', 'dept_id', 'item_id', 'sales', 'sell_price', 'event_name_1', 'snap_CA', 'snap_TX', 'snap_WI']
        view_cols = [c for c in view_cols if c in df.columns]

        return dash_table.DataTable(
            data=df[view_cols].head(200).to_dict('records'),
            columns=[{"name": c.replace('_', ' ').title(), "id": c} for c in view_cols],
            page_size=page_size or 25,
            style_header={'backgroundColor': '#0F172A', 'color': 'white', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'left', 'padding': '8px'},
            filter_action="native",
            sort_action="native"
        )
