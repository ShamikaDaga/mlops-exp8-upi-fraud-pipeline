"""Metric helpers: final metrics table, classification reports, financial impact."""
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)

from src.config import DEFAULT_THRESHOLD


def find_optimal_threshold(y_true, probabilities):
    """Optimise the fraud classification threshold using the F1 score."""
    precision_values, recall_values, thresholds = precision_recall_curve(y_true, probabilities)
    f1_values = (
        2 * precision_values[:-1] * recall_values[:-1]
        / (precision_values[:-1] + recall_values[:-1] + 1e-10)
    )
    best_threshold_index = f1_values.argmax()
    optimal_threshold = thresholds[best_threshold_index]
    print("Default threshold:", DEFAULT_THRESHOLD)
    print("Optimal threshold:", round(optimal_threshold, 4))
    print("Best F1 at optimal threshold:", round(f1_values[best_threshold_index], 4))
    return optimal_threshold, thresholds, f1_values


def print_classification_reports(y_true, y_pred_default, y_pred_optimal):
    print("=== DEFAULT THRESHOLD (0.50) ===")
    print(classification_report(
        y_true, y_pred_default, target_names=["Legitimate", "Fraud"], zero_division=0))
    print("\n=== OPTIMAL THRESHOLD ===")
    print(classification_report(
        y_true, y_pred_optimal, target_names=["Legitimate", "Fraud"], zero_division=0))


def final_metrics_table(y_true, y_pred_optimal, probabilities):
    """Final metrics at the optimal threshold."""
    final = {
        "Accuracy": accuracy_score(y_true, y_pred_optimal),
        "Precision": precision_score(y_true, y_pred_optimal, zero_division=0),
        "Recall": recall_score(y_true, y_pred_optimal, zero_division=0),
        "F1-Fraud": f1_score(y_true, y_pred_optimal, zero_division=0),
        "F1-Macro": f1_score(y_true, y_pred_optimal, average="macro", zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
    }
    table = pd.DataFrame({"Metric": list(final), "Score": list(final.values())})
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    return final, table


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def financial_impact(test_df, y_pred_optimal):
    """Rupee value of the fraud caught / missed and of the false alarms."""
    test_financial = test_df.copy()
    test_financial["Predicted_Fraud"] = y_pred_optimal
    # Total actual fraud amount
    total_fraud_amount = test_financial.loc[
        test_financial["Fraud_Flag"] == 1, "Transaction_Amount"].sum()
    # Fraud correctly identified
    fraud_prevented = test_financial.loc[
        (test_financial["Fraud_Flag"] == 1) & (test_financial["Predicted_Fraud"] == 1),
        "Transaction_Amount"].sum()
    # Fraud missed
    fraud_missed = total_fraud_amount - fraud_prevented
    # Legitimate transactions incorrectly flagged
    false_alarm_amount = test_financial.loc[
        (test_financial["Fraud_Flag"] == 0) & (test_financial["Predicted_Fraud"] == 1),
        "Transaction_Amount"].sum()
    fraud_capture_rate = fraud_prevented / total_fraud_amount * 100

    print(f"Total Actual Fraud Amount: ₹ {total_fraud_amount:,.2f}")
    print(f"Fraud Amount Correctly Flagged: ₹ {fraud_prevented:,.2f}")
    print(f"Fraud Amount Captured: {fraud_capture_rate:.2f}%")
    print(f"Fraud Amount Missed: ₹ {fraud_missed:,.2f}")
    print(f"Value of False Alarms: ₹ {false_alarm_amount:,.2f}")
    return fraud_capture_rate
