"""Post-training visualisations: comparison bars, threshold curve, confusion
matrix, feature importance and the decision-tree diagram."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, plot_tree  # noqa: E402

from eda.eda_plots import save_figure  # noqa: E402


def plot_model_comparison(leaderboard, metric, filename):
    """Bar chart of one metric for every model/method combination."""
    plt.figure(figsize=(12, 7))
    sns.barplot(data=leaderboard, x=metric, y="Model", hue="Method")
    plt.title(f"Model Comparison by {metric}")
    plt.xlabel(metric)
    plt.ylabel("Model")
    plt.legend(title="Training Method")
    plt.tight_layout()
    return save_figure(filename)


def plot_threshold_optimization(thresholds, f1_values, optimal_threshold, best_model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, f1_values)
    plt.axvline(0.50, linestyle="--", label="Default Threshold = 0.50")
    plt.axvline(optimal_threshold, linestyle="--", label=f"Optimal = {optimal_threshold:.3f}")
    plt.title(f"Threshold Optimization - {best_model_name}")
    plt.xlabel("Probability Threshold")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(alpha=0.2)
    return save_figure("model_threshold_optimization.png")


def plot_confusion_matrix(cm, best_model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"],
    )
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    return save_figure("model_confusion_matrix.png")


def compute_feature_importance(winning_model, feature_columns):
    """Tree importances or |coefficients|; None when the model has neither."""
    if hasattr(winning_model, "feature_importances_"):
        importance_values = winning_model.feature_importances_
    elif hasattr(winning_model, "coef_"):
        importance_values = np.abs(winning_model.coef_[0])
    else:
        importance_values = None
    if importance_values is None:
        print("Direct feature importance is not available for this model.")
        return None
    feature_importance_df = (
        pd.DataFrame({"Feature": feature_columns, "Importance": importance_values})
        .sort_values("Importance", ascending=False).reset_index(drop=True)
    )
    print("TOP 20 FRAUD DRIVERS")
    print(feature_importance_df.head(20).to_string())
    return feature_importance_df


def plot_feature_importance(feature_importance_df, best_model_name):
    top_features = feature_importance_df.head(15).sort_values("Importance")
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, x="Importance", y="Feature")
    plt.title(f"Top Fraud Drivers - {best_model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    return save_figure("model_feature_importance.png")


def plot_decision_tree(X_train_fs, y_train_fs):
    """Plot the Decision Tree (same settings as the notebook)."""
    decision_tree_model = DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42
    )
    decision_tree_model.fit(X_train_fs, y_train_fs)
    plt.figure(figsize=(24, 12))
    plot_tree(
        decision_tree_model, feature_names=X_train_fs.columns,
        class_names=["Legitimate", "Fraud"], filled=True, rounded=True, fontsize=8,
    )
    plt.title("Decision Tree Fraud Classification")
    return save_figure("model_decision_tree.png")
