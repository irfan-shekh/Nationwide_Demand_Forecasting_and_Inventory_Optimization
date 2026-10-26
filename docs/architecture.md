# System Architecture Documentation

## Nationwide Retail Demand Forecasting & Inventory Optimization System

This document outlines the end-to-end software and data science architecture of the Nationwide Demand Forecasting and Inventory Optimization platform.

```mermaid
flowchart TD
    subgraph Data Layer
        A1[Kaggle M5 Dataset] --> A2[Data Loader & Schema Validator]
        A2 --> A3[Type Downcaster & Memory Optimizer]
        A3 --> A4[Parquet Feature Cache]
    end

    subgraph Feature Engineering
        A4 --> B1[Calendar & SNAP Features]
        A4 --> B2[Price Dynamics & Discounts]
        A4 --> B3[Shift-Before-Roll Sales Lags & Statistics]
        B1 & B2 & B3 --> B4[Leakage-Safe Feature Store]
    end

    subgraph Multi-Model Forecasting Engine
        B4 --> C1[Model A: Last-Value & Seasonal-Naive 7D]
        B4 --> C2[Model B: Meta Prophet + Holidays]
        B4 --> C3[Model C: Global LightGBM + Quantiles]
        B4 --> C4[Model D: Temporal Fusion Transformer]
        C4 -.->|Colab Export Mode 2| C5[artifacts/tft/]
    end

    subgraph Hierarchical Reconciliation
        C1 & C2 & C3 & C5 --> D1[Bottom-Up Aggregation]
        C1 & C2 & C3 & C5 --> D2[MinT Optimal Shrinkage Projection]
        D1 & D2 --> D3[Coherent Hierarchy Predictions]
    end

    subgraph Operations Research Engine
        D3 --> E1[Safety Stock & ROP Calculator]
        E1 --> E2[Google OR-Tools MIP Solver SCIP]
        E2 --> E3[Constrained Replenishment Orders]
    end

    subgraph Presentation Layer
        D3 & E3 --> F1[Plotly Dash 7-Page Executive Dashboard]
        F1 --> G1[Page 1: Executive Overview]
        F1 --> G2[Page 2: Demand Forecasting]
        F1 --> G3[Page 3: Model Benchmark Comparison]
        F1 --> G4[Page 4: Interactive Inventory Optimization]
        F1 --> G5[Page 5: Hierarchical Consistency]
        F1 --> G6[Page 6: Interactive Data Explorer]
        F1 --> G7[Page 7: System & Model Provenance]
    end
```

---

## 1. Modular Architecture Overview

The system is organized into modular decoupled layers:

| Module / Package | Path | Responsibility |
| :--- | :--- | :--- |
| **Configuration** | `config/settings.py` | Centralized paths, forecast horizons, random seeds, and inventory defaults. |
| **Data Ingestion & Validation** | `src/data/` | CSV ingestion, downcasting (`reduce_mem_usage`), schema validation, missing/duplicate checks, and parquet caching. |
| **Feature Engineering** | `src/features/` | Calendar indicators, SNAP benefits, weekly price changes, and strict shift-before-rolling lag/rolling statistics. |
| **Forecasting Engine** | `src/models/` | Naive baselines, Prophet, LightGBM (point + quantiles), TFT Colab integration bridge, and Model Registry. |
| **Evaluation & Backtesting** | `src/evaluation/` | Chronological splitting (no shuffling), WAPE, RMSE, MAE, Signed Bias, and in-sample RMSSE. |
| **Uncertainty & Reconciliation** | `src/forecasting/` | 80%/95% interval evaluation, Bottom-Up summation, and MinT (Minimum Trace) shrinkage projection. |
| **Operations Research** | `src/optimization/` | Safety stock, Reorder Point, and Google OR-Tools Mixed-Integer Linear Programming (MIP) replenishment engine. |
| **Executive Dashboard** | `dashboard/`, `app.py` | Responsive 7-page Plotly Dash application with live what-if parameter re-optimization. |
| **Automated Test Suite** | `tests/` | 24 unit and integration tests verifying leakage prevention, metrics, reconciliation, and OR-Tools constraints. |
