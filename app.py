"""
Main Plotly Dash Application Entrypoint
Nationwide Retail Demand Forecasting & Inventory Optimization
"""
from pathlib import Path
import sys
import pandas as pd
from dash import Dash, html
import dash_bootstrap_components as dbc
import logging

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import SAMPLE_DATA_DIR
from dashboard.layout import create_header, create_tabs
from dashboard.callbacks import register_callbacks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load prepared datasets
merged_path = SAMPLE_DATA_DIR / "merged_sample_features.parquet"
pred_path = SAMPLE_DATA_DIR / "test_predictions.parquet"

if merged_path.exists():
    merged_df = pd.read_parquet(merged_path)
else:
    logger.warning(f"Merged sample not found at {merged_path}. Run prepare_data.py first.")
    merged_df = pd.DataFrame()

if pred_path.exists():
    pred_df = pd.read_parquet(pred_path)
else:
    logger.warning(f"Predictions not found at {pred_path}. Run train_models.py first.")
    pred_df = pd.DataFrame()

# Initialize Dash App with Bootstrap Theme
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Demand Forecasting & Inventory Optimization"
)
server = app.server

# Assemble Main Layout
app.layout = html.Div([
    create_header(),
    dbc.Container([
        create_tabs(),
        html.Div(id="tab-content-container", className="pb-5")
    ], fluid=True, style={"maxWidth": "1500px"})
], style={"backgroundColor": "#F8FAFC", "minHeight": "100vh"})

# Register callbacks
register_callbacks(app, merged_df, pred_df)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    logger.info(f"Starting Dash Web Application on http://0.0.0.0:{port} ...")
    app.run(debug=False, host="0.0.0.0", port=port)

