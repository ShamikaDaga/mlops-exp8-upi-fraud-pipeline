"""Prometheus instrumentation for the fraud-prediction service.

The metrics live in this module so that Streamlit's script re-runs do not
register them twice. `start_metrics_server()` exposes them on /metrics.
"""
import os
import threading

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, start_http_server

METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
_server_lock = threading.Lock()
_server_started = False


def _get_or_create(factory, name, *args, **kwargs):
    """Create a metric, or return the already-registered one (module reloads)."""
    try:
        return factory(name, *args, **kwargs)
    except ValueError:
        existing = REGISTRY._names_to_collectors.get(name)
        if existing is None:
            existing = REGISTRY._names_to_collectors[name + "_total"]
        return existing


REQUESTS = _get_or_create(
    Counter, "ml_prediction_requests_total",
    "Total prediction requests received", ["status"],
)
PREDICTIONS = _get_or_create(
    Counter, "ml_predictions_total",
    "Total predictions made, by predicted class", ["prediction_class"],
)
ERRORS = _get_or_create(
    Counter, "ml_errors_total",
    "Total errors raised while serving predictions", ["error_type"],
)
LATENCY = _get_or_create(
    Histogram, "ml_prediction_latency_seconds",
    "Time taken to produce one prediction",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
FRAUD_PROBABILITY = _get_or_create(
    Histogram, "ml_fraud_probability",
    "Distribution of predicted fraud probabilities",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
APP_UP = _get_or_create(
    Gauge, "ml_app_up", "1 when the model is loaded and the app is serving",
)
MODEL_LOADED = _get_or_create(
    Gauge, "ml_model_info", "Loaded model (value is always 1)", ["model", "method"],
)


def start_metrics_server(port=METRICS_PORT):
    """Start the /metrics HTTP endpoint once per process."""
    global _server_started
    with _server_lock:
        if not _server_started:
            start_http_server(port)
            _server_started = True
    return _server_started


def record_success(result, seconds):
    """Update every metric for one successful prediction."""
    REQUESTS.labels(status="success").inc()
    PREDICTIONS.labels(prediction_class=result["prediction"]).inc()
    FRAUD_PROBABILITY.observe(result["fraud_probability"])
    LATENCY.observe(seconds)


def record_error(error_type):
    """Update the metrics for one failed request."""
    REQUESTS.labels(status="error").inc()
    ERRORS.labels(error_type=error_type).inc()
