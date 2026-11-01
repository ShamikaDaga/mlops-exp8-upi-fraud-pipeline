# ---------- Stage 1: build the Python dependencies ----------
FROM python:3.14-slim AS builder
WORKDIR /build
COPY requirements-serving.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-serving.txt

# ---------- Stage 2: slim runtime image ----------
FROM python:3.14-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    METRICS_PORT=8000

# libgomp is required by LightGBM / XGBoost / CatBoost, curl for the healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
WORKDIR /app
COPY src ./src
COPY .streamlit ./.streamlit
COPY artifacts/serving ./artifacts/serving
COPY artifacts/plots ./artifacts/plots

# Streamlit UI on 8501, Prometheus /metrics on 8000
EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# serve.py starts the /metrics exporter at process start, then launches Streamlit
CMD ["python", "-m", "src.serve"]
