# UPI Fraud Detection - End-to-End MLOps Pipeline (Experiment 8)

A complete ML project (UPI payment fraud detection, 2,800 transactions) taken from notebook to a
monitored, containerised, orchestrated service:

| Stage | Tool |
|---|---|
| Version control | Git + GitHub |
| CI/CD | GitHub Actions (flake8 lint, pytest, Docker build + smoke test) |
| Experiment tracking + model registry | MLflow (`sqlite:///mlflow.db`) |
| Containers | Docker (multi-stage image) + Docker Compose |
| Orchestration | Kubernetes (Minikube): Deployment + NodePort Service |
| Observability | Prometheus + Grafana |
| Serving UI | Streamlit (+ `/metrics` endpoint) |

## Project layout

```
data/raw/            raw Excel dataset              data/processed/  feature store (generated)
src/                 config, data loading, feature engineering, preprocessing, ETL pipeline,
                     inference, Prometheus metrics, Streamlit app (app.py)
eda/                 eda_plots.py (EDA graphs), model_plots.py (post-training graphs)
models/              balancing.py (SMOTE / class weights), model_factory.py, train_models.py (+MLflow)
evaluation/          metrics.py, leaderboard.py, evaluate.py (threshold, reports, registry, export)
shap/                shap_explain.py (SHAP summary plots)   <- no __init__.py on purpose
tests/               pytest suite          k8s/  deployment + service      monitoring/  Prometheus + Grafana
artifacts/serving/   the slim model bundle baked into the Docker image
```

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py                       # ETL -> EDA -> train (14 runs) -> evaluate -> SHAP
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000
streamlit run src/app.py                     # UI http://localhost:8501 , metrics http://localhost:8000/metrics
pytest -v ; flake8
docker compose up --build -d                 # app 8501, Prometheus 9090, Grafana 3000 (admin/admin)
minikube start --driver=docker
minikube image build -t upi-fraud-app:latest .
kubectl apply -f k8s/ ; kubectl scale deployment upi-fraud-app --replicas=4
minikube service upi-fraud-service
```

## Models

Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, AdaBoost - each
trained with **Class Weights** and with **SMOTE** (14 MLflow runs), ranked by F1-Macro then ROC-AUC.
The winner is tuned to its optimal F1 threshold, explained with SHAP, registered in the MLflow
Model Registry as `UPI_Fraud_Detector` and exported to `artifacts/serving/` for the app.
