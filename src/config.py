"""Central configuration: paths, constants and MLflow settings.

Importing this module FIRST in every entry-point also applies the
Windows-friendly thread settings used in the original notebook.
"""
import os
import sys
import warnings
from pathlib import Path

# Thread settings for stable execution on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
# The notebook prints the rupee sign; make sure the console can show it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ------------------------------------------------------------------ paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_STORE_DIR = PROCESSED_DIR / "feature_store"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
SERVING_DIR = ARTIFACTS_DIR / "serving"

FILE_PATH = RAW_DATA_DIR / "PS2 - DataSet.xlsx"
SHEET_NAME = "UPI_Fraud_Transactions"

PREPROCESSOR_PATH = FEATURE_STORE_DIR / "preprocessor.joblib"
TRAIN_FEATURE_STORE_PATH = FEATURE_STORE_DIR / "upi_train_features.csv"
TEST_FEATURE_STORE_PATH = FEATURE_STORE_DIR / "upi_test_features.csv"
FEATURE_NAMES_PATH = FEATURE_STORE_DIR / "feature_names.csv"
CLEAN_ENGINEERED_PATH = FEATURE_STORE_DIR / "upi_clean_engineered_data.csv"

TRAINING_RESULTS_PATH = ARTIFACTS_DIR / "training_results.csv"
LEADERBOARD_PATH = ARTIFACTS_DIR / "model_leaderboard.csv"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"
FINAL_REPORT_PATH = ARTIFACTS_DIR / "final_model_report.csv"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"
TEST_SPLIT_PATH = PROCESSED_DIR / "test_raw_split.joblib"

# ------------------------------------------------------------- constants
TARGET = "Fraud_Flag"
RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_THRESHOLD = 0.50
COLUMNS_TO_DROP = [
    TARGET, "User_ID", "Transaction_ID",
    "Transaction_Date", "Transaction_Time", "Operating_System",
]

# ---------------------------------------------------------------- MLflow
# A database backend so the MLflow UI and the scripts share the SAME store.
# (Always run scripts from the project root so this relative path resolves.)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "UPI_Fraud_ETL_FeatureStore_Experiment"
REGISTERED_MODEL_NAME = "UPI_Fraud_Detector"


def ensure_directories():
    """Create every output directory used by the pipeline."""
    for directory in (
        RAW_DATA_DIR, PROCESSED_DIR, FEATURE_STORE_DIR,
        ARTIFACTS_DIR, MODELS_DIR, PLOTS_DIR, SERVING_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def setup_mlflow():
    """Point MLflow at the SQLite store and select the experiment."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print("MLflow Tracking URI:")
    print(mlflow.get_tracking_uri())
    print("\nMLflow Experiment:")
    print(EXPERIMENT_NAME)
    return mlflow
