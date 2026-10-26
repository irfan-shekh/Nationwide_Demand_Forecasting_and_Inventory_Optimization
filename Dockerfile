FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies for OR-Tools & LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and prepared sample data/models
COPY . .

# Expose port (Cloud Run, Render, Hugging Face, or standard 8050)
ENV PORT=8050
EXPOSE 8050

CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050", "--workers", "2", "--timeout", "120"]
