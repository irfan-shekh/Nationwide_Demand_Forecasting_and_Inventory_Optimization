# Engineering Reflection

**Author:** Senior Data Science Intern  
**Project:** Nationwide Demand Forecasting and Inventory Optimization  
**Date:** October 2026  

---

## 1. Technical Skills Learned & Deepened
1. **Large-Scale Time-Series Feature Engineering:**
   Deepened practical mastery of constructing leakage-safe rolling statistics, lag matrices, and calendar covariates under non-stationary retail conditions. Learned how subtle indexing bugs can introduce lookahead leakage and how strict shift-before-rolling unit tests ensure mathematical integrity.
2. **Global Gradient Boosting vs Classical Time Series:**
   Observed firsthand the dramatic performance advantage of global tabular models (LightGBM) trained across thousands of disparate retail series compared to univariate classical models (Prophet/ARIMA). The global model learns cross-series price elasticities and category dynamics that individual models fail to capture.
3. **Deep Sequence-to-Sequence Modeling (TFT):**
   Implemented Temporal Fusion Transformer architectures utilizing variable selection networks, gating mechanisms, and multi-head attention to model complex multi-horizon interactions.
4. **Hierarchical Reconciliation Mathematics:**
   Implemented the structural summing matrix $S$ and derived the MinT (Minimum Trace) projection matrix $P = (S^T W^{-1} S)^{-1} S^T W^{-1}$ with Ledoit-Wolf shrinkage error covariance to ensure numerical stability.
5. **Constrained Operations Research with Google OR-Tools:**
   Formulated real-world Mixed-Integer Linear Programs (MIP) incorporating semi-continuous minimum order quantity (MOQ) binary triggers, warehouse shelf capacity bounds, and capital budget constraints solved via the SCIP branch-and-cut solver.
6. **Production Application Engineering:**
   Built a multi-page interactive Plotly Dash dashboard with decoupled data layers, modular callbacks, and real-time parameter what-if re-optimization.

---

## 2. Components Personally Implemented
- Ingestion and schema validation pipeline with automated RAM downcasting (`src/data/loader.py`, `validation.py`, `preprocessing.py`).
- Leakage-safe feature extraction engine with automated leakage unit tests (`src/features/engineering.py`).
- Benchmark models: Last-Value Naive, Seasonal-Naive 7-Day, Meta Prophet, and LightGBM point & quantile regressors (`src/models/`).
- Self-contained 13-cell Google Colab Temporal Fusion Transformer workflow (`notebooks/colab_tft_training.ipynb`) and artifact bridge (`src/models/deep_learning_model.py`).
- Retail hierarchy reconciliation engine (Bottom-Up and MinT) with consistency checkers (`src/forecasting/reconciliation.py`).
- Operations research inventory replenishment engine (`src/optimization/inventory_policy.py`, `ortools_optimizer.py`).
- Interactive 7-page Plotly Dash dashboard (`dashboard/layout.py`, `callbacks.py`, `charts.py`, `app.py`).
- Complete automated test suite comprising 24 unit and integration tests (`tests/`).

---

## 3. Problems Solved
- **58M Row Memory Bottleneck:** Solved via integer downcasting and creating a stratified representative development sample preserving all states, stores, and product hierarchies.
- **Timestamp Downcasting Collision:** Identified and resolved a type comparison bug in `reduce_mem_usage` when encountering `Timestamp` objects.
- **Zero-Division in Metrics & Budget:** Resolved division-by-zero edge cases in WAPE when actual demand is zero and in budget utilization when budget is set to zero.
- **Singular Covariance Matrix in MinT:** Implemented Ledoit-Wolf shrinkage regularization to ensure positive-definiteness when inverting large residual covariance matrices.
- **Semi-Continuous MOQ Formulation in OR-Tools:** Formulated big-M binary indicator constraints to prevent the solver from placing infeasible micro-orders below supplier minimum batch requirements.

---

## 4. Challenges Encountered & Limitations Discovered
- **Intermittent Zero Demand:** Over 63% of daily observations are zero. Standard mean squared error objectives tend to under-forecast peaks. Quantile regression and Poisson/Tweedie objectives were necessary to mitigate bias.
- **Deep Learning Dependency Weights:** Running heavy deep learning libraries like PyTorch Forecasting locally on lightweight CPU machines creates substantial overhead; decoupling TFT to Google Colab while maintaining Mode 2 artifact compatibility in the main repo proved essential.
- **Deterministic Lead Time Assumption:** Supplier lead times were modeled as deterministic; real supply chains exhibit stochastic lead-time variability.

---

## 5. Future Improvements
- **Stochastic Lead Time Modeling:** Incorporate convolution of demand distribution and lead time distribution ($SS = z \sqrt{L \sigma_D^2 + D^2 \sigma_L^2}$).
- **Multi-Echelon Network Optimization:** Extend the single-store optimizer into a two-echelon distribution center to retail store replenishment network with transshipment routing.
- **Price Elasticity & Promotion Optimization:** Jointly optimize retail selling price and order quantity using non-linear demand response curves.
