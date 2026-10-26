# Nationwide Retail Demand Forecasting & Inventory Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 24 Passed](https://img.shields.io/badge/Tests-24%20Passed-brightgreen.svg)]()
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7.svg?logo=render&logoColor=white)](https://nationwide-demand-forecasting-and.onrender.com)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Dashboard: Plotly Dash](https://img.shields.io/badge/dashboard-Plotly%20Dash-orange.svg)](https://dash.plotly.com/)
[![OR-Tools: 9.15](https://img.shields.io/badge/optimization-Google%20OR--Tools-blueviolet.svg)](https://developers.google.com/optimization)

> 🌐 **Live Web Application**: [https://nationwide-demand-forecasting-and.onrender.com](https://nationwide-demand-forecasting-and.onrender.com)

An enterprise-grade, end-to-end Machine Learning, Hierarchical Time-Series Forecasting, and Operations Research system built for the Walmart M5 retail dataset. Forecasts multi-tier store demand, estimates multi-quantile uncertainty intervals, reconciles predictions across the national retail hierarchy, and generates cost-optimal replenishment recommendations using Google OR-Tools Mixed-Integer Linear Programming (MIP).


---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Model Benchmark Results](#-model-benchmark-results)
- [Project Directory Structure](#-project-directory-structure)
- [Quickstart & Setup](#-quickstart--setup)
- [Running the Complete Pipeline](#-running-the-complete-pipeline)
- [Interactive Dashboard](#-interactive-dashboard)
- [Google Colab Deep Learning (TFT) Workflow](#-google-colab-deep-learning-tft-workflow)
- [Automated Testing Suite](#-automated-testing-suite)
- [Documentation & Deliverables](#-documentation--deliverables)

---

## 🚀 Key Features

1. **Leakage-Safe Feature Engineering**:
   - Calendar features: Day of week, month, year, weekend indicators, US holidays, and state-specific SNAP benefit flags.
   - Dynamic price features: Normalization vs historical item mean, week-over-week price differential, and percentage discount.
   - Shift-before-rolling historical statistics: Strictly shifts sales by $\ge 1$ (or 28 for direct multi-step) before rolling means and deviations to guarantee zero future information leakage.
2. **Multi-Model Forecasting Engine**:
   - **Model A (Baselines)**: Last-Value Persistence and 7-Day Seasonal-Naive benchmarks.
   - **Model B (Prophet)**: Additive trend and seasonal decomposition with US holiday regressors.
   - **Model C (LightGBM)**: Global multi-store gradient booster with multi-quantile uncertainty estimation ($\alpha = 0.1, 0.5, 0.9$).
   - **Model D (Temporal Fusion Transformer)**: Self-contained Google Colab GPU workflow (`notebooks/colab_tft_training.ipynb`) with multi-head attention and quantile loss.
3. **Hierarchical Reconciliation**:
   - Supports 6-level retail hierarchy: $\text{National} \rightarrow \text{State} \rightarrow \text{Store} \rightarrow \text{Department} \rightarrow \text{Category} \rightarrow \text{Item-Store}$.
   - **Bottom-Up Reconciler**: Exact additive aggregation ($S \cdot y_{\text{bottom}} = y_{\text{all}}$).
   - **MinT (Minimum Trace) Reconciler**: Optimal linear projection $P = (S^T W^{-1} S)^{-1} S^T W^{-1}$ with Ledoit-Wolf shrinkage error covariance.
4. **Google OR-Tools Constrained Inventory Optimization**:
   - Solves multi-item replenishment Mixed-Integer Linear Program (SCIP branch-and-cut).
   - Minimizes: Holding Costs + Stockout Shortage Penalties + Fixed Transaction Costs.
   - Respects: Semi-continuous Minimum Order Quantity (MOQ), inventory balance, store warehouse shelf capacity, and procurement budget constraints.
5. **Plotly Dash 7-Page Executive Dashboard**:
   - Executive Overview, Demand Forecasting Explorer, Model Benchmark, Inventory Optimizer What-If Sensitivity, Hierarchical Reconciliation Consistency, Data Explorer, and Model Provenance Status.
6. **Automated Unit & Integration Testing**:
   - 24 tests in `pytest` covering schema validation, feature leakage, evaluation metrics, prediction intervals, reconciliation, and OR-Tools optimization.

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[M5 Retail Data] --> B[Validation & Loader]
    B --> C[Leakage-Safe Feature Store]
    C --> D[Multi-Model Engine]
    D --> E[Reconciliation Engine]
    E --> F[OR-Tools Replenishment Solver]
    E & F --> G[Plotly Dash Executive Dashboard]
```

---

## 🏆 Model Benchmark Results

All models evaluated on the **exact same 28-day chronological holdout period** (2016-03-28 to 2016-04-24) across 3,360 item-store observations:

| Rank | Model | Model Paradigm | Test WAPE | Test RMSE | Signed Bias | RMSSE (Scaled) | 80% Interval Coverage | Avg Width | Training Time |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Temporal Fusion Transformer (TFT)** | Deep Attention Seq2Seq | **1.0096** | 1.0301 | **+0.0036** | **0.4986** | **92.0%** | 1.87 units | 42.1s (GPU) |
| 🥈 | **LightGBM** | Global Gradient Boosting | **1.0159** | **1.0280** | +0.0299 | 0.4976 | **90.5%** | 1.78 units | **1.83s** |
| 🥉 | **Prophet** | Additive Seasonality | 1.2864 | 3.2239 | +0.8073 | 1.3089 | 50.5% | 3.06 units | 8.80s |
| 4 | **Seasonal-Naive (7-Day)** | Baseline Cycle | 1.3095 | 1.5565 | +0.0750 | 0.7534 | - | - | 0.05s |
| 5 | **Naive (Last-Value)** | Baseline Persistence | 1.5284 | 1.8542 | +0.2750 | 0.8975 | - | - | 0.02s |

*Key Finding: Global LightGBM and Deep TFT models outperform naive baselines by **over 34 percentage points in WAPE** while maintaining high interval reliability.*

---

## 📁 Project Directory Structure

```text
Nationwide Demand Forecasting and Inventory Optimization/
├── app.py                         # Plotly Dash interactive dashboard entrypoint
├── requirements.txt               # Pinned reproducible dependencies
├── README.md                      # Primary project documentation
├── .gitignore                     # Git protection for large CSVs and binary checkpoints
├── config/
│   └── settings.py                # Centralized project configuration and defaults
├── src/
│   ├── data/
│   │   ├── loader.py              # Ingestion, memory downcasting, and sampling
│   │   ├── validation.py          # Schema validation, missing/duplicate checks
│   │   └── preprocessing.py       # Wide-to-long melting and covariate joining
│   ├── features/
│   │   └── engineering.py         # Leakage-safe calendar, price, and rolling features
│   ├── models/
│   │   ├── baselines.py           # Naive and Seasonal-Naive models
│   │   ├── lightgbm_model.py      # LightGBM point and quantile regressors
│   │   ├── prophet_model.py       # Meta Prophet forecaster with holiday modeling
│   │   ├── deep_learning_model.py # TFT Colab artifact loader (Modes 1 & 2)
│   │   └── model_registry.py      # Unified model orchestration facade
│   ├── evaluation/
│   │   ├── metrics.py             # WAPE, RMSE, MAE, Bias, and RMSSE
│   │   └── backtesting.py         # Chronological non-shuffled splitting
│   ├── forecasting/
│   │   ├── intervals.py           # Prediction interval coverage and width metrics
│   │   └── reconciliation.py      # Bottom-Up and MinT optimal reconciliation
│   └── optimization/
│       ├── inventory_policy.py    # Safety stock, ROP, and inventory cost formulas
│       └── ortools_optimizer.py   # Google OR-Tools MIP replenishment solver
├── dashboard/
│   ├── layout.py                  # 7-page Bootstrap responsive layout definitions
│   ├── callbacks.py               # Reactive Dash callbacks and live re-optimizer
│   └── charts.py                  # High-definition Plotly visualization generators
├── notebooks/
│   ├── eda.ipynb                  # Exploratory Data Analysis notebook
│   └── colab_tft_training.ipynb   # 13-cell self-contained Google Colab TFT notebook
├── scripts/
│   ├── prepare_data.py            # Sample extraction and data quality reporting
│   ├── train_models.py            # Model training and holdout benchmarking
│   ├── evaluate_models.py         # Hierarchical evaluation & MinT verification
│   └── run_pipeline.py            # Master end-to-end execution script
├── tests/
│   ├── test_data_validation.py    # Schema, downcasting, and null checks
│   ├── test_features.py           # Feature extraction & strict leakage prevention
│   ├── test_metrics.py            # Metric formulas and zero-denominator handling
│   ├── test_forecasting.py        # Model inference and interval bounds validity
│   ├── test_reconciliation.py     # Bottom-up and MinT aggregation coherence
│   └── test_inventory_optimizer.py# Safety stock, ROP, and OR-Tools MIP constraints
├── artifacts/tft/                 # Exported Colab TFT checkpoints, configs, and predictions
├── models/                        # Serialized production model artifacts
├── reports/                       # Machine-readable evaluation reports (JSON & CSV)
└── docs/
    ├── architecture.md            # Detailed system design and flowcharts
    ├── methodology.md             # Mathematical formulations and derivations
    ├── deployment.md              # Cloud deployment and containerization guide
    ├── presentation_outline.md    # 10-minute presentation outline
    └── reflection.md              # Personal engineering reflection
```

---

## ⚙️ Quickstart & Setup

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/nationwide-demand-forecasting.git
cd nationwide-demand-forecasting

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Dataset Setup
The project uses the official Kaggle M5 Forecasting Competition dataset:
- Place raw CSV files (`calendar.csv`, `sales_train_validation.csv`, `sales_train_evaluation.csv`, `sell_prices.csv`, and `sample_submission.csv`) in the [`data/raw/`](data/raw/) directory.
- Centralized path resolution in [`config/settings.py`](config/settings.py) automatically resolves from `data/raw/` (with fallback to project root).

---

## 🏃 Running the Complete Pipeline

Execute the master pipeline to prepare data, train models, evaluate hierarchies, and optimize inventory:
```bash
# Windows:
.venv\Scripts\python.exe scripts/run_pipeline.py

# Linux/macOS:
python scripts/run_pipeline.py
```

Or execute individual modular stages:
```bash
# Step 1: Prepare stratified benchmark data and quality report
.venv\Scripts\python.exe scripts/prepare_data.py

# Step 2: Train Naive, Seasonal-Naive, LightGBM, and Prophet models
.venv\Scripts\python.exe scripts/train_models.py

# Step 3: Evaluate retail hierarchy and verify MinT reconciliation
.venv\Scripts\python.exe scripts/evaluate_models.py
```

---

## 🖥️ Interactive Dashboard

Launch the responsive Plotly Dash web dashboard locally:
```bash
# Windows (direct via venv):
.venv\Scripts\python.exe app.py

# Or after activating your virtual environment:
# .venv\Scripts\activate
# python app.py
```
Open **[http://127.0.0.1:8050](http://127.0.0.1:8050)** in any modern web browser.

### Available Dashboard Pages:
1. **📊 Executive Overview**: National demand trajectories, state sales share, top items, and key inventory alerts.
2. **🔮 Demand Forecasting**: Dynamic series-level historical demand, forecasts, 80% confidence bands, and provenance metadata.
3. **🏆 Model Benchmark**: Interactive comparison of WAPE, RMSE, MAE, Bias, and training latencies.
4. **📦 Inventory Optimization**: Real-time what-if parameter sliders (lead time, service level, budget, shelf capacity) triggering live Google OR-Tools re-optimization.
5. **🏛️ Hierarchical Analysis**: Multi-tier retail consistency verification across National, State, Store, Department, and Category tiers.
6. **🔍 Data Explorer**: Paginated, filterable tabular exploration of sales, price, calendar events, and SNAP indicators.
7. **⚙️ System & Model Status**: Live model artifact status, runtime configurations, and provenance tracking.

---

## ☁️ Google Colab Deep Learning (TFT) Workflow

A verified, executed 13-cell notebook with cell outputs is located at:
[`notebooks/colab_tft_training.ipynb`](notebooks/colab_tft_training.ipynb)

- **Environment**: Executed on free Google Colab **Tesla T4 GPU** (14.56 GB VRAM).
- **Results**: Achieved **0.8886 Test WAPE** and **90.3% empirical interval coverage** on 28-day holdout predictions.
- **Features**: GPU detection, PyTorch Forecasting `TimeSeriesDataSet` with `NaNLabelEncoder(add_nan=True)`, multi-head attention TFT training, baseline comparison, and automated artifact export.
- **Integration**: Artifacts exported to `/content/artifacts/tft/` integrate directly into the main repository ([`artifacts/tft/`](artifacts/tft/)), allowing the local dashboard to operate without an active Colab session.

---

## 🧪 Automated Testing Suite

Run the full automated test suite with `pytest`:
```bash
pytest -v
```
**Results: 24 passed in ~3.3 seconds (100% pass rate).**

Test coverage includes:
- `test_data_validation.py`: Schema validation, missing values, duplicates, downcasting.
- `test_features.py`: Calendar features, price dynamics, and strict temporal leakage prevention.
- `test_metrics.py`: WAPE, RMSE, MAE, Signed Bias, and RMSSE edge cases (zero denominators, negative clipping).
- `test_forecasting.py`: Baseline predictions, LightGBM inference, and interval ordering ($lower \le upper$).
- `test_reconciliation.py`: Exact bottom-up summation and MinT summing matrix $S$ coherence.
- `test_inventory_optimizer.py`: Safety stock, ROP, semi-continuous MOQ constraints, warehouse capacity, and zero-budget fallbacks.

---

## 📚 Documentation & Deliverables

- [System Architecture](docs/architecture.md)
- [Methodology & Mathematical Formulations](docs/methodology.md)
- [Deployment & Containerization Guide](docs/deployment.md)
- [10-Minute Presentation Outline](docs/presentation_outline.md)
- [Engineering Reflection](docs/reflection.md)

---

## 📄 License
This project is licensed under the MIT License.
