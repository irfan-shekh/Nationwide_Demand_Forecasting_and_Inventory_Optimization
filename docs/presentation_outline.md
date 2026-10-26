# 10-Minute Presentation Outline

## Nationwide Retail Demand Forecasting & Inventory Optimization
**Presenter:** Senior Data Science Intern  
**Audience:** Executive Leadership, Supply Chain Directors, and Senior Engineering Panel  
**Duration:** 10 Minutes (8 mins presentation + 2 mins Q&A)

---

### Slide 1: Executive Title & Business Problem (0:00 - 1:00)
- **Title:** End-to-End Retail Demand Forecasting & Inventory Replenishment Optimization.
- **The Core Business Problem:**
  - Retailers lose billions annually to simultaneous stockouts (lost revenue, customer churn) and over-stocking (working capital lockup, storage costs, obsolescence).
  - Volatile daily retail sales exhibit severe intermittency (>60% zero-demand days), weekly surges, calendar events, and price sensitivity across national store networks.
- **Project Mission:**
  - Build a production-grade system that forecasts store-level item demand, quantifies uncertainty intervals, reconciles predictions across the retail tree, and uses Google OR-Tools to solve optimal replenishment orders subject to physical warehouse capacity and capital budgets.

---

### Slide 2: Dataset Characteristics & Data Quality Insights (1:00 - 2:00)
- **Dataset:** Walmart M5 Forecasting competition data (1,913 historical daily sales across 10 stores in CA, TX, WI).
- **Data Engineering Realities:**
  - 30,490 series melted across 1,913 days yields 58+ million records.
  - Type downcasting (`reduce_mem_usage`) reduced RAM by 72% (from int64 to int16/float32).
  - Zero-sales rate: 63.3% across items. Intermittent demand classified via Syntetos-Boylan / Croston metrics (ADI > 1.32).
  - Automated schema validation verified 0 negative sales and 0 duplicate keys.

---

### Slide 3: Leakage-Safe Feature Engineering Pipeline (2:00 - 3:00)
- **The Golden Rule:** At prediction date $t$, no information from $t$ or $t + k$ is accessible.
- **Shift-Before-Rolling Formulation:**
  - Sales series are strictly shifted by $\ge 1$ (or $H=28$ for direct multi-step) before computing 7-day, 14-day, and 28-day rolling means and standard deviations.
  - Verified by automated unit tests corrupting future sales without altering current features.
- **Engineered Covariates:**
  - Calendar: Day of week, weekend indicator, month, holiday flags, SNAP food stamp benefits by state.
  - Price: Normalized selling price, week-over-week price change, relative discount percentage.

---

### Slide 4: Multi-Model Benchmark Comparison (3:00 - 4:30)
- **Evaluation Discipline:** All models evaluated on identical 28-day chronological holdout period (2016-03-28 to 2016-04-24). No shuffled splits.
- **Genuine Benchmark Results:**

| Model | Architecture | Test WAPE | Test RMSE | Signed Bias | Scaled RMSSE | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Temporal Fusion Transformer (TFT)** | Multi-Head Attention Seq2Seq | **1.0096** | 1.0301 | **+0.0036** | **0.4986** | 42.1s (GPU) |
| **LightGBM** | Global Gradient Boosting + Quantiles | **1.0159** | **1.0280** | +0.0299 | 0.4976 | **1.83s** |
| **Prophet** | Additive Decomposition + Holidays | 1.2864 | 3.2239 | +0.8073 | 1.3089 | 8.80s |
| **Seasonal-Naive** | 7-Day Weekly Cycle Baseline | 1.3095 | 1.5565 | +0.0750 | 0.7534 | 0.05s |
| **Naive Baseline** | Last-Value Persistence | 1.5284 | 1.8542 | +0.2750 | 0.8975 | 0.02s |

- **Takeaway:** Advanced machine learning reduces forecast error (WAPE) by **over 34 percentage points** relative to naive baselines.

---

### Slide 5: Uncertainty Quantification & Colab TFT Workflow (4:30 - 5:30)
- **Beyond Point Forecasts:** Retailers cannot plan inventory with single-number guesses.
- **Prediction Intervals:**
  - LightGBM multi-quantile regression ($\alpha = 0.1, 0.5, 0.9$) and TFT quantile loss generate calibrated 80% and 95% confidence bounds.
  - Empirical 80% interval coverage: **90.5% (LightGBM)** and **92.0% (TFT)** with average interval width under 1.88 units.
- **Colab Decoupled Deep Learning:**
  - Executable 13-cell notebook (`notebooks/colab_tft_training.ipynb`) handles GPU acceleration, model checkpointing, and artifact export (`artifacts/tft/`), ensuring main deployment does not require an active Colab instance.

---

### Slide 6: Hierarchical Forecast Reconciliation (5:30 - 6:30)
- **Hierarchy:** National $\rightarrow$ State $\rightarrow$ Store $\rightarrow$ Department $\rightarrow$ Category $\rightarrow$ Item-Store.
- **Reconciliation Engine:**
  - Bottom-Up aggregation: Lower-tier predictions exactly equal higher-tier totals ($S \cdot y_{\text{bottom}} = y_{\text{all}}$ with 0.00 discrepancy).
  - MinT (Minimum Trace) optimal projection: Linear projection matrix $P = (S^T W^{-1} S)^{-1} S^T W^{-1}$ with Ledoit-Wolf shrinkage error covariance $W$.
  - Automated tests verify structural consistency across all levels.

---

### Slide 7: Operations Research & Google OR-Tools Engine (6:30 - 7:45)
- **Bridging Data Science to Supply Chain Execution:**
  - Forecasts feed directly into mathematical optimization.
- **Constrained Replenishment Formulation:**
  - Minimize: $\sum_i (h_i \cdot I_{\text{end}, i} + p_i \cdot \text{Shortage}_i + K_i \cdot z_i)$
  - Subject to: Semi-continuous MOQ ($Q_i \ge \text{MOQ} \cdot z_i$), inventory conservation, warehouse capacity, and procurement budget constraints.
- **OR-Tools SCIP Output:**
  - Recommended order: 1,760.5 units ($4,036.39 spend). Solved to exact optimality (`STATUS: OPTIMAL`).
  - Budget utilization: 16.1% (within $25,000 capital limit).

---

### Slide 8: Interactive Plotly Dash Application Demo (7:45 - 8:45)
- **Page Walkthrough:**
  1. *Executive Overview:* National demand trajectories, state sales share, top item volumes, and KPI alerts.
  2. *Demand Forecasting:* Interactive series explorer with 80% confidence bands and model provenance badges.
  3. *Model Comparison:* Dynamic WAPE/RMSE metric rankings and training latency bars.
  4. *Inventory Optimization:* Live what-if sensitivity sliders (lead time, service level, budget) re-solving OR-Tools in real-time.
  5. *Hierarchy & Data Explorer:* Aggregation verification and paginated data table.

---

### Slide 9: Challenges Overcome, Limitations & Next Steps (8:45 - 9:30)
- **Technical Challenges Solved:**
  - 58M row unpivoting mitigated by vectorized downcasting and representative stratified sampling.
  - Zero-division in metrics and optimizer edge cases guarded.
  - Numerical singular matrix in MinT regularized via shrinkage covariance.
- **Future Improvements:**
  - Cross-store cannibalization and promotional price elasticity coefficients.
  - Multi-echelon distribution center to retail store replenishment network routing.

---

### Slide 10: Conclusion & Q&A (9:30 - 10:00)
- **Deliverable Summary:**
  - GitHub-ready repository, 24 passing automated tests, self-contained Colab notebook, trained LightGBM artifacts, OR-Tools optimizer, and interactive Plotly Dash app.
- **Questions & Discussion.**
