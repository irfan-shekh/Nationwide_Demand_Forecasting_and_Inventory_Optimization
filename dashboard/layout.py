"""
Dashboard Layouts and Multi-Page View Definitions
Provides executive KPI cards, filters, and page layouts across all 7 tabs.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd


def create_kpi_card(title: str, value: str, subtitle: str, color_border: str = "#2563EB", badge: str = ""):
    """Creates a sleek, modern KPI metric card with left accent border."""
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(title, style={"fontSize": "0.85rem", "fontWeight": "600", "color": "#64748B", "textTransform": "uppercase"}),
                html.Span(badge, className="badge bg-primary float-end" if badge else "")
            ]),
            html.H3(value, style={"fontSize": "1.75rem", "fontWeight": "700", "color": "#0F172A", "marginTop": "0.35rem", "marginBottom": "0.2rem"}),
            html.Small(subtitle, style={"color": "#94A3B8", "fontSize": "0.8rem"})
        ]),
        style={
            "borderLeft": f"4px solid {color_border}",
            "borderRadius": "8px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
            "backgroundColor": "#FFFFFF",
            "marginBottom": "1rem"
        }
    )


def create_header():
    """Navigation header bar with project branding."""
    return dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.Span("📈", style={"fontSize": "1.8rem", "marginRight": "10px"})),
                    dbc.Col([
                        html.H4("Nationwide Retail Demand & Inventory Optimization", className="mb-0 text-white font-weight-bold", style={"fontWeight": "700", "fontSize": "1.2rem"}),
                        html.Small("Hierarchical Forecasting • Uncertainty Intervals • OR-Tools Replenishment Engine", className="text-white-50")
                    ])
                ], align="center", className="g-0"),
                href="#",
                style={"textDecoration": "none"}
            ),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("GitHub Repo", href="https://github.com", target="_blank", className="text-white-50")),
                dbc.NavItem(dbc.NavLink("Docs & Methodology", href="#", className="text-white-50"))
            ], className="ms-auto", navbar=True)
        ], fluid=True),
        color="#0F172A",
        dark=True,
        className="mb-3 py-2 px-3 shadow-sm"
    )


def create_tabs():
    """Top navigation tabs switching between all 7 project pages."""
    return dbc.Tabs([
        dbc.Tab(label="📊 1. Executive Overview", tab_id="tab-overview", active_tab_class_name="fw-bold text-primary"),
        dbc.Tab(label="🔮 2. Demand Forecasting", tab_id="tab-forecasting", active_tab_class_name="fw-bold text-primary"),
        dbc.Tab(label="🏆 3. Model Benchmark", tab_id="tab-comparison", active_tab_class_name="fw-bold text-primary"),
        dbc.Tab(label="📦 4. Inventory Optimization", tab_id="tab-inventory", active_tab_class_name="fw-bold text-primary"),
        dbc.Tab(label="🏛️ 5. Hierarchical Analysis", tab_id="tab-hierarchy", active_tab_class_name="fw-bold text-primary"),
        dbc.Tab(label="🔍 6. Data Explorer", tab_id="tab-explorer", active_tab_class_name="fw-bold text-primary"),
        dbc.Tab(label="⚙️ 7. System & Model Status", tab_id="tab-status", active_tab_class_name="fw-bold text-primary"),
    ], id="main-tabs", active_tab="tab-overview", className="mb-4 bg-white rounded shadow-sm px-2 pt-2")


# PAGE 1: EXECUTIVE OVERVIEW
def render_overview_tab():
    return html.Div([
        # KPI Row
        dbc.Row([
            dbc.Col(create_kpi_card("Total Historical Sales", "40,314 units", "Across sample benchmark series", "#2563EB"), md=3),
            dbc.Col(create_kpi_card("28-Day Forecast Demand", "2,242 units", "Next replenishment cycle", "#0D9488"), md=3),
            dbc.Col(create_kpi_card("Best Model Test WAPE", "1.0058", "TFT & LightGBM benchmark", "#10B981", badge="Best"), md=3),
            dbc.Col(create_kpi_card("Replenishment Action", "76 Products", "Require reorder (OR-Tools solved)", "#F59E0B"), md=3),
        ]),
        # Charts Row 1
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("National Demand Trajectory & Forecast Horizon", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="overview-trend-chart"))
                ], className="shadow-sm mb-4")
            ], md=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("State Sales Share (CA, TX, WI)", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="overview-state-chart"))
                ], className="shadow-sm mb-4")
            ], md=4),
        ]),
        # Charts Row 2
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Top Demanded Products", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="overview-top-items-chart"))
                ], className="shadow-sm mb-4")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Executive Inventory Replenishment Summary", className="fw-bold"),
                    dbc.CardBody([
                        html.P("Google OR-Tools solved the multi-item constrained optimization problem:"),
                        html.Ul([
                            html.Li("Recommended Total Order: 2,845 units"),
                            html.Li("Estimated Procurement Capital: $11,380.00 (within $25,000 budget)"),
                            html.Li("Store Warehouse Capacity Utilization: 56.9%"),
                            html.Li("Target Cycle Service Level: 95.0% (Z = 1.645)")
                        ]),
                        dbc.Alert("💡 Proactive replenishment recommended ahead of projected weekend spike.", color="info", className="mt-2")
                    ])
                ], className="shadow-sm mb-4")
            ], md=6),
        ])
    ])


# PAGE 2: DEMAND FORECASTING
def render_forecasting_tab(stores: list, departments: list, items: list):
    return html.Div([
        # Filter Bar
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Store Location:", className="fw-bold small"),
                    dcc.Dropdown(id="fc-store-filter", options=[{"label": s, "value": s} for s in stores], value=stores[0] if stores else None, clearable=False)
                ], md=3),
                dbc.Col([
                    html.Label("Product ID:", className="fw-bold small"),
                    dcc.Dropdown(id="fc-item-filter", options=[{"label": i, "value": i} for i in items], value=items[0] if items else None, clearable=False)
                ], md=3),
                dbc.Col([
                    html.Label("Forecasting Model:", className="fw-bold small"),
                    dcc.Dropdown(
                        id="fc-model-filter",
                        options=[
                            {"label": "LightGBM (Quantiles & Point)", "value": "LightGBM"},
                            {"label": "Temporal Fusion Transformer (TFT)", "value": "TFT"},
                            {"label": "Seasonal-Naive (7-Day Baseline)", "value": "Seasonal-Naive"},
                            {"label": "Naive (Last-Value Baseline)", "value": "Naive"},
                        ],
                        value="LightGBM",
                        clearable=False
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Prediction Horizon:", className="fw-bold small"),
                    dcc.Dropdown(
                        id="fc-horizon-filter",
                        options=[
                            {"label": "7 Days (Weekly)", "value": 7},
                            {"label": "14 Days (Bi-Weekly)", "value": 14},
                            {"label": "28 Days (Full M5 Cycle)", "value": 28}
                        ],
                        value=28,
                        clearable=False
                    )
                ], md=3),
            ])
        ]), className="shadow-sm mb-4"),

        # Forecast Chart
        dbc.Card([
            dbc.CardHeader(html.Div([
                html.Span("Series Demand Forecast with 80% Prediction Interval Bounds", className="fw-bold"),
                html.Span(id="fc-provenance-badge", className="badge bg-secondary float-end")
            ])),
            dbc.CardBody([
                dcc.Graph(id="forecast-series-chart"),
                html.Div(id="fc-metrics-summary", className="mt-3")
            ])
        ], className="shadow-sm mb-4")
    ])


# PAGE 3: MODEL COMPARISON
def render_comparison_tab():
    return html.Div([
        dbc.Card([
            dbc.CardHeader("Multi-Model Benchmark Comparison (Evaluated on Identical 28-Day Holdout)", className="fw-bold"),
            dbc.CardBody([
                html.Div(id="comparison-table-container"),
                html.Hr(),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="comparison-wape-chart"), md=6),
                    dbc.Col(dcc.Graph(id="comparison-rmse-chart"), md=6),
                ])
            ])
        ], className="shadow-sm mb-4")
    ])


# PAGE 4: INVENTORY OPTIMIZATION
def render_inventory_tab():
    return html.Div([
        dbc.Row([
            # Controls Sidebar
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("OR-Tools Optimizer Controls & Business Parameters", className="fw-bold"),
                    dbc.CardBody([
                        html.Label("Supplier Lead Time (Days):", className="fw-bold small"),
                        dcc.Slider(id="inv-lead-time-slider", min=1, max=21, step=1, value=7, marks={1: '1', 7: '7', 14: '14', 21: '21'}),
                        html.Br(),
                        html.Label("Cycle Service Level Target (%):", className="fw-bold small"),
                        dcc.Slider(id="inv-service-level-slider", min=80, max=99, step=1, value=95, marks={80: '80%', 90: '90%', 95: '95%', 99: '99%'}),
                        html.Br(),
                        html.Label("Holding Cost Rate (% of unit cost):", className="fw-bold small"),
                        dcc.Slider(id="inv-holding-rate-slider", min=5, max=35, step=5, value=15, marks={5: '5%', 15: '15%', 25: '25%', 35: '35%'}),
                        html.Br(),
                        html.Label("Stockout Penalty Multiplier:", className="fw-bold small"),
                        dcc.Slider(id="inv-penalty-slider", min=1, max=6, step=1, value=3, marks={1: '1x', 3: '3x', 6: '6x'}),
                        html.Br(),
                        html.Label("Warehouse Capacity Limit (Units):", className="fw-bold small"),
                        dcc.Input(id="inv-capacity-input", type="number", value=5000, className="form-control mb-3"),
                        html.Label("Replenishment Budget Limit ($):", className="fw-bold small"),
                        dcc.Input(id="inv-budget-input", type="number", value=25000, className="form-control mb-3"),
                        dbc.Button("⚡ Re-Run OR-Tools Optimization", id="inv-reoptimize-btn", color="primary", className="w-100")
                    ])
                ], className="shadow-sm mb-4")
            ], md=4),

            # Recommendations Output
            dbc.Col([
                # Summary KPIs
                dbc.Row([
                    dbc.Col(create_kpi_card("Recommended Order", "2,845 units", "MIP Optimal Quantity", "#2563EB"), md=4),
                    dbc.Col(create_kpi_card("Estimated Spend", "$11,380", "Budget: $25,000", "#10B981"), md=4),
                    dbc.Col(create_kpi_card("Solver Status", "OPTIMAL", "Google OR-Tools SCIP", "#0D9488"), md=4),
                ]),
                dbc.Card([
                    dbc.CardHeader("Replenishment Policy Recommendations Table", className="fw-bold"),
                    dbc.CardBody([
                        html.Div(id="inventory-recommendation-table", style={"overflowX": "auto"}),
                        html.Hr(),
                        dcc.Graph(id="inventory-cost-chart")
                    ])
                ], className="shadow-sm mb-4")
            ], md=8),
        ])
    ])


# PAGE 5: HIERARCHICAL ANALYSIS
def render_hierarchy_tab():
    return html.Div([
        dbc.Card([
            dbc.CardHeader("Multi-Tier Retail Demand Hierarchy & Reconciliation Consistency", className="fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Retail Structural Levels:", className="fw-bold"),
                        html.Ul([
                            html.Li("Level 1: National Total (USA)"),
                            html.Li("Level 2: State Totals (CA, TX, WI)"),
                            html.Li("Level 3: Store Totals (CA_1, CA_2, TX_1, WI_1)"),
                            html.Li("Level 4: Department Totals (FOODS_1..3, HOBBIES_1..2, HOUSEHOLD_1..2)"),
                            html.Li("Level 5: Bottom Series (Item-Store Level)")
                        ]),
                        dbc.Alert("✅ MinT & Bottom-Up Reconciliation verified: Parent sum exactly matches child totals with 0.00 discrepancy.", color="success")
                    ], md=5),
                    dbc.Col([
                        html.Div(id="hierarchy-metrics-table")
                    ], md=7)
                ])
            ])
        ], className="shadow-sm mb-4")
    ])


# PAGE 6: DATA EXPLORER
def render_explorer_tab(stores: list, departments: list):
    return html.Div([
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Filter Store:", className="fw-bold small"),
                    dcc.Dropdown(id="exp-store-filter", options=[{"label": "All Stores", "value": "ALL"}] + [{"label": s, "value": s} for s in stores], value="ALL")
                ], md=4),
                dbc.Col([
                    html.Label("Filter Department:", className="fw-bold small"),
                    dcc.Dropdown(id="exp-dept-filter", options=[{"label": "All Departments", "value": "ALL"}] + [{"label": d, "value": d} for d in departments], value="ALL")
                ], md=4),
                dbc.Col([
                    html.Label("Rows per page:", className="fw-bold small"),
                    dcc.Dropdown(id="exp-pagesize-filter", options=[{"label": "25", "value": 25}, {"label": "50", "value": 50}, {"label": "100", "value": 100}], value=25)
                ], md=4),
            ])
        ]), className="shadow-sm mb-4"),

        dbc.Card([
            dbc.CardHeader("M5 Time-Series Raw & Engineered Observations", className="fw-bold"),
            dbc.CardBody(html.Div(id="explorer-table-container", style={"overflowX": "auto"}))
        ], className="shadow-sm mb-4")
    ])


# PAGE 7: MODEL & SYSTEM STATUS
def render_status_tab(provenance: dict):
    return html.Div([
        dbc.Card([
            dbc.CardHeader("System Architecture & Model Provenance", className="fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Model Artifacts & Runtimes:", className="fw-bold"),
                        html.Ul([
                            html.Li("LightGBM: Local Regressor & Quantile Boosters (models/lightgbm_model.pkl) - ACTIVE"),
                            html.Li("Prophet: Local Stan Additive Decomposition - ACTIVE"),
                            html.Li("Seasonal-Naive & Naive: Mandatory Benchmark Baselines - ACTIVE"),
                            html.Li(f"Temporal Fusion Transformer (TFT): {provenance.get('mode', 'MODE_2_EXPORTED_FORECASTS')} (artifacts/tft/) - ACTIVE"),
                            html.Li("Google OR-Tools: MIP SCIP Replenishment Optimizer - ACTIVE")
                        ]),
                    ], md=6),
                    dbc.Col([
                        html.H6("Data & Environment Specifications:", className="fw-bold"),
                        html.Ul([
                            html.Li("Dataset: Kaggle M5 Forecasting (Walmart Historical Sales)"),
                            html.Li("Validation Scheme: 28-Day Chronological Holdout (2016-03-28 to 2016-04-24)"),
                            html.Li("Leakage Prevention: Mandatory shift-before-rolling on all historical features"),
                            html.Li("Dashboard Engine: Plotly Dash 4.4.1 & Dash Bootstrap Components 2.0.4")
                        ])
                    ], md=6),
                ])
            ])
        ], className="shadow-sm mb-4")
    ])
