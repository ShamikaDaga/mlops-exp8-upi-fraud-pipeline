"""Container entrypoint: start the Prometheus exporter, then run the Streamlit UI.

Streamlit only executes src/app.py once a browser session connects, so the
/metrics endpoint must be started here, at process start, to be scrapeable
immediately (Prometheus, CI smoke test, Kubernetes probes).

    python -m src.serve
"""
import json
import os
import sys

from src import metrics
from src.config import ROOT_DIR, SERVING_DIR

PORT = os.getenv("STREAMLIT_PORT", "8501")


def main():
    with open(SERVING_DIR / "model_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    metrics.start_metrics_server()
    metrics.APP_UP.set(1)
    metrics.MODEL_LOADED.labels(
        model=meta["best_model"], method=meta["training_method"]).set(1)
    print(f"Prometheus metrics on :{metrics.METRICS_PORT}/metrics", flush=True)

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", str(ROOT_DIR / "src" / "app.py"),
        f"--server.port={PORT}", "--server.address=0.0.0.0",
        "--server.headless=true", "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
