"""Train every model with both imbalance-handling methods and track in MLflow.

Run from the project root:   python -m models.train_models
"""
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from src import config
from src.config import (
    DEFAULT_THRESHOLD, MODELS_DIR, RANDOM_STATE, TEST_SIZE,
    TRAIN_FEATURE_STORE_PATH, TRAINING_RESULTS_PATH,
)
from src.preprocessing import load_feature_store

from models.balancing import apply_smote, calculate_scale_pos_weight
from models.model_factory import METHODS, MODEL_PARAMETERS, create_models


def model_artifact_path(model_name, method):
    """artifacts/models/<Model_Name>_<Method>.joblib"""
    safe_model_name = model_name.replace(" ", "_").replace("/", "_")
    return MODELS_DIR / f"{safe_model_name}_{method.replace(' ', '_')}.joblib"


def train_and_track():
    config.ensure_directories()
    mlflow = config.setup_mlflow()

    X_train_fs, y_train_fs, X_test_fs, y_test_fs = load_feature_store()
    print("\nTraining feature shape:", X_train_fs.shape)
    print("Testing feature shape:", X_test_fs.shape)

    print("\n=== SMOTE ===")
    X_train_smote, y_train_smote = apply_smote(X_train_fs, y_train_fs)
    print("\n=== CLASS / SCALE WEIGHT ===")
    scale_pos_weight = calculate_scale_pos_weight(y_train_fs)

    results = []
    for method in METHODS:
        print("METHOD:", method)
        if method == "Class Weights":
            X_train_current, y_train_current = X_train_fs, y_train_fs
        else:
            X_train_current, y_train_current = X_train_smote, y_train_smote

        models = create_models(method, scale_pos_weight)
        for model_name, model in models.items():
            print(f"\nTraining {model_name} using {method}...")
            # Train
            model.fit(X_train_current, y_train_current)

            # Predictions
            probabilities = model.predict_proba(X_test_fs)[:, 1]
            predictions = (probabilities >= DEFAULT_THRESHOLD).astype(int)

            # Metrics
            accuracy = accuracy_score(y_test_fs, predictions)
            precision = precision_score(y_test_fs, predictions, zero_division=0)
            recall = recall_score(y_test_fs, predictions, zero_division=0)
            f1_fraud = f1_score(y_test_fs, predictions, zero_division=0)
            f1_macro = f1_score(y_test_fs, predictions, average="macro", zero_division=0)
            roc_auc = roc_auc_score(y_test_fs, probabilities)
            pr_auc = average_precision_score(y_test_fs, probabilities)

            results.append({
                "Model": model_name, "Method": method, "Accuracy": accuracy,
                "Precision": precision, "Recall": recall, "F1-Fraud": f1_fraud,
                "F1-Macro": f1_macro, "ROC-AUC": roc_auc, "PR-AUC": pr_auc,
            })

            # MLFLOW RUN
            run_name = model_name.replace(" ", "_") + "_" + method.replace(" ", "_")
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("model", model_name)
                mlflow.set_tag("imbalance_method", method)
                mlflow.set_tag("dataset", "UPI Fraud Dataset")
                mlflow.log_param("model", model_name)
                mlflow.log_param("imbalance_method", method)
                mlflow.log_param("test_size", TEST_SIZE)
                mlflow.log_param("random_state", RANDOM_STATE)
                mlflow.log_param("feature_count", X_train_fs.shape[1])
                # Model-specific parameters
                for param_name, param_value in MODEL_PARAMETERS[model_name].items():
                    mlflow.log_param(param_name, param_value)
                # Metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_fraud", f1_fraud)
                mlflow.log_metric("f1_macro", f1_macro)
                mlflow.log_metric("roc_auc", roc_auc)
                mlflow.log_metric("pr_auc", pr_auc)
                # Save model artifact
                model_path = model_artifact_path(model_name, method)
                joblib.dump(model, model_path)
                mlflow.log_artifact(str(model_path), artifact_path="model")
                # Log feature store
                mlflow.log_artifact(str(TRAIN_FEATURE_STORE_PATH), artifact_path="feature_store")
                print(
                    f"Accuracy={accuracy:.4f} | F1={f1_fraud:.4f} | "
                    f"F1-Macro={f1_macro:.4f} | ROC-AUC={roc_auc:.4f}"
                )
    print("\nAll model experiments completed.")
    results_df = pd.DataFrame(results)
    results_df.to_csv(TRAINING_RESULTS_PATH, index=False)
    print("Training results saved at:", TRAINING_RESULTS_PATH)
    return results_df


if __name__ == "__main__":
    train_and_track()
