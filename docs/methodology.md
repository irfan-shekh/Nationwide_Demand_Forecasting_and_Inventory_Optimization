# Methodology & Mathematical Formulations

This document provides detailed mathematical formulations, statistical principles, and operations research methodologies implemented in this project.

---

## 1. Leakage Prevention Principles

In retail demand forecasting, temporal data leakage occurs when future sales or test-period information inadvertently contaminates training features.

### Shift-Before-Rolling Rule
For any observation at date $t$, rolling features (such as 7-day, 14-day, or 28-day moving averages and standard deviations) are computed **strictly after shifting historical sales**:
$$\text{ShiftedSales}_{i, t} = \text{Sales}_{i, t - \text{shift}}$$
$$\text{RollingMean}_{i, t}^{(w)} = \frac{1}{w} \sum_{k=0}^{w-1} \text{Sales}_{i, t - \text{shift} - k}$$
where $\text{shift} \ge 1$ for recursive multi-step forecasting or $\text{shift} = H = 28$ for direct multi-step forecasting.

---

## 2. Evaluation Metrics

All models are evaluated on identical holdout dates using standard retail metrics:

1. **Weighted Absolute Percentage Error (WAPE):**
   $$\text{WAPE} = \frac{\sum_{t=1}^H |y_t - \hat{y}_t|}{\sum_{t=1}^H |y_t|}$$
   Unlike MAPE, WAPE avoids division by zero on zero-sales days and weights high-volume items proportionally.

2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{H} \sum_{t=1}^H (y_t - \hat{y}_t)^2}$$

3. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{H} \sum_{t=1}^H |y_t - \hat{y}_t|$$

4. **Signed Forecast Bias:**
   $$\text{Bias} = \frac{1}{H} \sum_{t=1}^H (\hat{y}_t - y_t)$$
   - $\text{Bias} > 0$: Systematic over-forecasting (excess holding cost risk).
   - $\text{Bias} < 0$: Systematic under-forecasting (stockout risk).

5. **Root Mean Squared Scaled Error (RMSSE):**
   $$\text{RMSSE} = \sqrt{\frac{\frac{1}{H} \sum_{t=1}^H (y_t - \hat{y}_t)^2}{\frac{1}{N-7} \sum_{t=8}^N (y_t - y_{t-7})^2}}$$
   Scale-independent benchmark metric scaled by the in-sample seasonal-naive difference.

---

## 3. Hierarchical Reconciliation

Retail demand forms a strict hierarchy:
$$\text{National} \rightarrow \text{State} \rightarrow \text{Store} \rightarrow \text{Department} \rightarrow \text{Category} \rightarrow \text{Item-Store}$$

### Summing Matrix $S$
Let $y_t$ be the vector of all $m$ series at time $t$, and $b_t$ be the vector of $n_b$ bottom-level series. The structural summing matrix $S \in \{0, 1\}^{m \times n_b}$ enforces:
$$y_t = S b_t$$

### MinT (Minimum Trace) Optimal Projection
MinT finds linear unbiased reconciled forecasts $\tilde{y}_t$ minimizing the total forecast error variance:
$$\tilde{y}_t = S (S^T W^{-1} S)^{-1} S^T W^{-1} \hat{y}_t$$
where:
- $\hat{y}_t$ is the base forecast vector from individual models.
- $W$ is the forecast error covariance matrix.
- To guarantee numerical stability, we use a Ledoit-Wolf shrinkage estimator:
  $$W_{\text{shrink}} = (1 - \lambda) W_{\text{sample}} + \lambda \text{diag}(W_{\text{sample}}) + \epsilon I$$

---

## 4. Google OR-Tools Inventory Replenishment Formulation

The inventory optimizer formulates replenishment as a Mixed-Integer Linear Program (MIP):

### Objective Function
Minimize total expected holding, shortage, and ordering transaction costs:
$$\min \sum_{i=1}^n \left( h_i \cdot I_{end, i} + p_i \cdot \text{Shortage}_i + K_i \cdot z_i \right)$$

### Subject to:
1. **Inventory Flow Conservation:**
   $$I_{end, i} - \text{Shortage}_i = I_{0, i} + Q_i - D_i \quad \forall i$$
2. **Semi-Continuous Minimum Order Quantity (MOQ):**
   $$Q_i \ge \text{MOQ}_i \cdot z_i \quad \forall i$$
   $$Q_i \le M \cdot z_i \quad \forall i, \quad z_i \in \{0, 1\}$$
3. **Store Warehouse Shelf Capacity:**
   $$\sum_{i=1}^n (I_{0, i} + Q_i) \le \text{Capacity}$$
4. **Capital Replenishment Budget:**
   $$\sum_{i=1}^n (c_i \cdot Q_i) \le \text{Budget}$$
5. **Non-Negativity:**
   $$Q_i \ge 0, \quad I_{end, i} \ge 0, \quad \text{Shortage}_i \ge 0$$
