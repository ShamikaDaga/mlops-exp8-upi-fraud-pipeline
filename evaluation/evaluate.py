"""Evaluation stage: leaderboard, threshold, reports, plots, MLflow + serving export.

Run from the project root:   python -m evaluation.evaluate
"""
import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from src import config
from src.config import (
    ARTIFACTS_DIR, DEFAULT_THRESHOLD, FEATURE_IMPORTANCE_PATH, FINAL_REPORT_PATH,
    LEADERBOARD_PATH, PLOTS_DIR, PREDICTIONS_PATH, REGISTERED_MODEL_NAME,
    SERVING_DIR, TEST_SPLIT_PATH,
)
from src.preprocessing import load_feature_store

from eda.model_plots import (
    compute_feature_importance, plot_confusion_matrix, plot_decision_tree,
    plot_feature_importance, plot_model_comparison, plot_threshold_optimization,
)
from evaluation.leaderboard import build_leaderboard, identify_best_model, save_leaderboard
from evaluation.metrics import (
    confusion, final_metrics_table, financial_impact, find_optimal_threshold,
    print_classification_reports,
)


def _log_registered_model(mlflow, winning_model, best_model_name, X_sample):
    """Log the winning model to MLflow and register it in the Model Registry."""
    try:
        info = mlflow.sklearn.log_model(
            winning_model, name="best_model",
            input_example=X_sample.head(3),
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="cloudpickle",
        )
        print("Registered model:", REGISTERED_MODEL_NAME, "->", info.model_uri)
    except Exception as exc:  # registry logging must never break the pipeline
        print("Model registry logging skipped:", exc)


def run_evaluation():
    config.ensure_directories()
    mlflow = config.setup_mlflow()
    import mlflow.sklearn  # noqa: F401  (flavour needed for log_model)

    X_train_fs, y_train_fs, X_test_fs, y_test_fs = load_feature_store()
    test_df = joblib.load(TEST_SPLIT_PATH)

    print("\n=== LEADERBOARD ===")
    leaderboard = build_leaderboard()
    save_leaderboard(leaderboard)
    best_model_name, best_method, best_row, winning_model = identify_best_model(leaderboard)

    print("\n=== COMPARISON PLOTS ===")
    plot_model_comparison(leaderboard, "F1-Macro", "model_comparison_f1_macro.png")
    plot_model_comparison(leaderboard, "ROC-AUC", "model_comparison_roc_auc.png")

    print("\n=== THRESHOLD OPTIMISATION ===")
    best_probabilities = winning_model.predict_proba(X_test_fs)[:, 1]
    optimal_threshold, thresholds, f1_values = find_optimal_threshold(
        y_test_fs, best_probabilities)
    plot_threshold_optimization(thresholds, f1_values, optimal_threshold, best_model_name)

    print("\n=== EVALUATION AT OPTIMAL THRESHOLD ===")
    y_pred_default = (best_probabilities >= DEFAULT_THRESHOLD).astype(int)
    y_pred_optimal = (best_probabilities >= optimal_threshold).astype(int)
    print_classification_reports(y_test_fs, y_pred_default, y_pred_optimal)

    print("\n=== CONFUSION MATRIX ===")
    cm = confusion(y_test_fs, y_pred_optimal)
    print(cm)
    plot_confusion_matrix(cm, best_model_name)

    print("\n=== FINAL METRICS ===")
    final, _ = final_metrics_table(y_test_fs, y_pred_optimal, best_probabilities)

    print("\n=== FINANCIAL IMPACT ===")
    fraud_capture_rate = financial_impact(test_df, y_pred_optimal)

    print("\n=== FEATURE IMPORTANCE ===")
    feature_importance_df = compute_feature_importance(winning_model, X_train_fs.columns)
    if feature_importance_df is not None:
        plot_feature_importance(feature_importance_df, best_model_name)

    print("\n=== DECISION TREE PLOT ===")
    plot_decision_tree(X_train_fs, y_train_fs)

    print("\n=== PREDICTIONS ===")
    predictions = pd.DataFrame({
        "Transaction_ID": test_df["Transaction_ID"].values,
        "Fraud_Flag_Pred": y_pred_optimal,
        "Fraud_Flag_Prob": best_probabilities,
    })
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print("Predictions saved at:", PREDICTIONS_PATH)
    print(predictions.head(10).to_string())

    print("\n=== FINAL MODEL ANALYSIS REPORT ===")
    final_report = pd.DataFrame({
        "Item": [
            "Best Model", "Training Method", "Optimal Threshold", "Accuracy",
            "Precision", "Recall", "F1-Fraud", "F1-Macro", "ROC-AUC", "PR-AUC",
            "Fraud Capture Rate",
        ],
        "Value": [
            best_model_name, best_method, optimal_threshold, final["Accuracy"],
            final["Precision"], final["Recall"], final["F1-Fraud"],
            final["F1-Macro"], final["ROC-AUC"], final["PR-AUC"],
            fraud_capture_rate / 100,
        ],
    })
    print(final_report.to_string())
    final_report.to_csv(FINAL_REPORT_PATH, index=False)

    print("\n=== MLFLOW: FINAL ANALYSIS ===")
    with mlflow.start_run(run_name="Best_Model_Final_Analysis"):
        mlflow.set_tag("stage", "Final_Model_Analysis")
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_param("training_method", best_method)
        mlflow.log_param("optimal_threshold", float(optimal_threshold))
        mlflow.log_metric("accuracy_optimal_threshold", final["Accuracy"])
        mlflow.log_metric("precision_optimal_threshold", final["Precision"])
        mlflow.log_metric("recall_optimal_threshold", final["Recall"])
        mlflow.log_metric("f1_fraud_optimal_threshold", final["F1-Fraud"])
        mlflow.log_metric("f1_macro_optimal_threshold", final["F1-Macro"])
        mlflow.log_metric("roc_auc", final["ROC-AUC"])
        mlflow.log_metric("pr_auc", final["PR-AUC"])
        mlflow.log_metric("fraud_capture_rate", fraud_capture_rate)
        mlflow.log_artifact(str(LEADERBOARD_PATH), artifact_path="analysis")
        mlflow.log_artifact(str(PREDICTIONS_PATH), artifact_path="analysis")
        mlflow.log_artifact(str(FINAL_REPORT_PATH), artifact_path="analysis")
        if feature_importance_df is not None:
            feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
            mlflow.log_artifact(str(FEATURE_IMPORTANCE_PATH), artifact_path="analysis")
        # Plots produced by the eda/ scripts
        for plot in sorted(PLOTS_DIR.glob("*.png")):
            mlflow.log_artifact(str(plot), artifact_path="plots")
        # Model Registry
        _log_registered_model(mlflow, winning_model, best_model_name, X_test_fs)
    print("Final model analysis tracked in MLflow.")

    print("\n=== EXPORT SERVING ARTIFACTS ===")
    joblib.dump(winning_model, SERVING_DIR / "best_model.joblib")
    leaderboard.to_csv(SERVING_DIR / "model_leaderboard.csv", index=False)
    final_report.to_csv(SERVING_DIR / "final_model_report.csv", index=False)
    meta = {
        "best_model": best_model_name,
        "training_method": best_method,
        "optimal_threshold": float(optimal_threshold),
        "metrics": {k: float(v) for k, v in final.items()},
        "fraud_capture_rate": float(fraud_capture_rate),
        "feature_count": int(X_train_fs.shape[1]),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(SERVING_DIR / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Serving artefacts written to:", SERVING_DIR)

    print("\nFINAL EXPERIMENT VERIFICATION")
    print("Models tested:", leaderboard["Model"].nunique())
    print("Total MLflow model runs:", len(leaderboard))
    print("Best model:", best_model_name, "|", best_method)
    print("ROC-AUC:", round(final["ROC-AUC"], 4))
    print("F1-Macro:", round(final["F1-Macro"], 4))
    print("F1-Fraud:", round(final["F1-Fraud"], 4))
    print("Optimal Threshold:", round(optimal_threshold, 4))
    print("Artifacts folder:", ARTIFACTS_DIR)


if __name__ == "__main__":
    run_evaluation()
