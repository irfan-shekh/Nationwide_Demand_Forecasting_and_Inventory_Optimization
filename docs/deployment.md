# Deployment & Operations Guide

This guide details how to run, containerize, and deploy the Nationwide Demand Forecasting and Inventory Optimization system locally and in the cloud.

---

## 1. Local Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- 4GB+ RAM recommended
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/nationwide-demand-forecasting.git
cd nationwide-demand-forecasting

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Running the End-to-End Pipeline
```bash
# Run data preparation, training, hierarchical evaluation, and OR-Tools optimization
python scripts/run_pipeline.py
```

### Launching the Interactive Web Dashboard
```bash
python app.py
```
Open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your web browser.

---

## 2. Running Automated Tests
```bash
pytest -v
```
All 24 unit and integration tests validate schema conformity, leakage prevention, metric formulas, prediction intervals, MinT reconciliation, and OR-Tools constraints.

---

## 3. Google Colab Temporal Fusion Transformer (TFT) Workflow

The deep learning model is fully decoupled from local runtime constraints:
1. Open Google Colab: [https://colab.research.google.com](https://colab.research.google.com).
2. Upload `notebooks/colab_tft_training.ipynb`.
3. Set Runtime to **T4 GPU** (`Runtime` > `Change runtime type` > `T4 GPU`).
4. Execute cells 1 through 13.
5. Download exported files from `/content/artifacts/tft/` into `artifacts/tft/`:
   - `model.ckpt`
   - `predictions.csv`
   - `model_config.json`
   - `evaluation_metrics.json`
   - `training_metadata.json`
6. The dashboard automatically detects these artifacts and switches to **Mode 2 (Colab Exported Forecasts)** or **Mode 1 (Live Checkpoint Inference)**.

---

## 4. Containerized Cloud Deployment (Docker)

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies for OR-Tools & LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8050

CMD ["gunicorn", "-b", "0.0.0.0:8050", "--workers", "2", "app:server"]
```

### Build & Run Docker Container
```bash
docker build -t demand-forecast-app .
docker run -p 8050:8050 demand-forecast-app
```

---

## 5. Lightweight Production Hosting Considerations
- **Data Footprint**: The raw M5 dataset is over 400MB. In production, only the processed parquet sample (`data/sample/*.parquet`), serialized models (`models/*.pkl`), and TFT predictions (`artifacts/tft/predictions.csv`) need to be bundled with the web image.
- **Stateless Inference**: The Dash app runs statelessly on Gunicorn / Uvicorn, enabling seamless horizontal autoscaling on Google Cloud Run, AWS App Runner, or Render.
