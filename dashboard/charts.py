"""
Dashboard Visualizations and Plotly Chart Generators
Creates publication-quality, interactive, responsive charts with curated palettes.
"""
from typing import Dict, Any, Optional, List
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Curated Professional Color Palette
PRIMARY_COLOR = "#2563EB"    # Royal Blue
SECONDARY_COLOR = "#0D9488"  # Teal
ACCENT_COLOR = "#F59E0B"     # Amber / Warning
DANGER_COLOR = "#EF4444"     # Crimson / Shortage
SUCCESS_COLOR = "#10B981"    # Emerald / In-Stock
DARK_BG = "#0F172A"
CARD_BG = "#FFFFFF"
TEXT_COLOR = "#1E293B"


def create_demand_forecast_chart(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    title: str = "Demand Forecast vs Actuals with 80% Prediction Interval",
    show_actuals: bool = True
) -> go.Figure:
    """
    Plots historical actuals, holdout actuals, point predictions, and shaded prediction interval bounds.
    """
    fig = go.Figure()

    # 1. Historical Actuals (Solid Slate Line)
    if not historical_df.empty:
        fig.add_trace(go.Scatter(
            x=historical_df['date'],
            y=historical_df['sales'],
            mode='lines',
            name='Historical Demand',
            line=dict(color='#64748B', width=2),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual Sales:</b> %{y:.1f} units<extra></extra>'
        ))

    # 2. Test Period Actuals (if available and requested)
    if show_actuals and 'actual' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['actual'],
            mode='lines+markers',
            name='Actual Holdout Sales',
            line=dict(color='#0F172A', width=2.5),
            marker=dict(size=4),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual:</b> %{y:.1f} units<extra></extra>'
        ))

    # 3. Prediction Intervals (Shaded Area)
    if 'pred_upper' in forecast_df.columns and 'pred_lower' in forecast_df.columns:
        # Upper bound trace
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['pred_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Lower bound trace with fill
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['pred_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(37, 99, 235, 0.18)',
            name='80% Prediction Interval',
            hovertemplate='<b>Upper:</b> %{y:.1f}<extra></extra>'
        ))

    # 4. Point Forecast (Vibrant Blue Dashed Line)
    if 'predicted' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['predicted'],
            mode='lines+markers',
            name='Forecast (Point)',
            line=dict(color=PRIMARY_COLOR, width=3, dash='dash'),
            marker=dict(size=5, symbol='diamond'),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Forecast:</b> %{y:.1f} units<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR, family='Inter, sans-serif')),
        xaxis=dict(title='Date', gridcolor='#E2E8F0', showgrid=True),
        yaxis=dict(title='Daily Sales (Units)', gridcolor='#E2E8F0', showgrid=True),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode='x unified'
    )
    return fig


def create_model_comparison_bar(comparison_df: pd.DataFrame, metric: str = 'wape') -> go.Figure:
    """Creates benchmark bar chart comparing models on chosen metric."""
    df_sorted = comparison_df.sort_values(by=metric, ascending=True)

    metric_labels = {
        'wape': 'WAPE (Lower is Better)',
        'rmse': 'RMSE (Lower is Better)',
        'mae': 'MAE (Lower is Better)',
        'rmsse': 'RMSSE (Scaled Error)',
        'bias': 'Signed Bias (Closer to 0 is Better)'
    }

    colors = [
        PRIMARY_COLOR if 'LightGBM' in m or 'TFT' in m else '#94A3B8'
        for m in df_sorted['Model']
    ]

    fig = go.Figure(go.Bar(
        x=df_sorted[metric],
        y=df_sorted['Model'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:.4f}" for v in df_sorted[metric]],
        textposition='outside',
        hovertemplate='<b>Model:</b> %{y}<br><b>' + metric.upper() + ':</b> %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=f"Model Performance: {metric_labels.get(metric, metric.upper())}", font=dict(size=15)),
        xaxis=dict(title=metric.upper(), gridcolor='#E2E8F0'),
        yaxis=dict(autorange='reversed'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=50, b=40)
    )
    return fig


def create_state_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """State-level sales volume comparison chart."""
    state_agg = df.groupby('state_id')['sales'].sum().reset_index()
    state_names = {'CA': 'California', 'TX': 'Texas', 'WI': 'Wisconsin'}
    state_agg['state_name'] = state_agg['state_id'].map(state_names).fillna(state_agg['state_id'])

    fig = px.bar(
        state_agg,
        x='state_name',
        y='sales',
        color='state_name',
        color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR],
        title="Historical Sales Volume by State",
        labels={'sales': 'Total Units Sold', 'state_name': 'State'}
    )
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', showlegend=False)
    return fig


def create_top_products_chart(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart for top N selling products."""
    top_items = df.groupby('item_id')['sales'].sum().nlargest(top_n).reset_index()
    fig = px.bar(
        top_items.sort_values('sales'),
        x='sales',
        y='item_id',
        orientation='h',
        title=f"Top {top_n} Products by Sales Volume",
        labels={'sales': 'Units Sold', 'item_id': 'Product ID'},
        color_discrete_sequence=[SECONDARY_COLOR]
    )
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    return fig


def create_inventory_cost_breakdown(recs_df: pd.DataFrame) -> go.Figure:
    """Pie chart showing holding vs shortage vs order procurement cost shares."""
    if recs_df.empty:
        return go.Figure()

    holding = recs_df['holding_cost'].sum()
    shortage = recs_df['shortage_cost'].sum()
    order = recs_df['order_cost'].sum()

    fig = go.Figure(go.Pie(
        labels=['Holding Cost', 'Shortage Cost', 'Procurement Cost'],
        values=[holding, shortage, order],
        hole=0.45,
        marker=dict(colors=[ACCENT_COLOR, DANGER_COLOR, PRIMARY_COLOR]),
        textinfo='label+percent',
        hovertemplate='<b>%{label}:</b> $%{value:,.2f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text="Total Replenishment Cost Composition", font=dict(size=15)),
        paper_bgcolor='white',
        margin=dict(l=30, r=30, t=50, b=30)
    )
    return fig
