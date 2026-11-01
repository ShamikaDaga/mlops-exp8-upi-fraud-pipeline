"""SHAP explainability for the best model.

Run from the project root:   python shap/shap_explain.py

NOTE: this folder must NOT contain an __init__.py, otherwise it would shadow
the real `shap` library on import.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from src import config  # noqa: E402
from src.config import LEADERBOARD_PATH, PLOTS_DIR  # noqa: E402
from src.preprocessing import load_feature_store  # noqa: E402
from evaluation.leaderboard import identify_best_model  # noqa: E402


def generate_shap_summary(winning_model, best_model_name, X_train_fs, X_test_fs):
    """SHAP summary plot (beeswarm) for the winning model."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Use a subset so SHAP remains fast
        shap_sample = X_test_fs.iloc[:min(300, len(X_test_fs))]
        if best_model_name in [
            "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "AdaBoost",
        ]:
            explainer = shap.TreeExplainer(winning_model)
            shap_values = explainer.shap_values(shap_sample)
            # Some versions/models return a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            explainer = shap.LinearExplainer(winning_model, X_train_fs)
            shap_values = explainer.shap_values(shap_sample)
        plt.figure()
        shap.summary_plot(
            shap_values, shap_sample, feature_names=X_test_fs.columns, show=False)
        plt.title(f"SHAP Summary - {best_model_name}")
        plt.tight_layout()
        path = PLOTS_DIR / "shap_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved plot:", path)

        # Mean |SHAP| bar chart (companion global-importance view)
        plt.figure()
        shap.summary_plot(
            shap_values, shap_sample, feature_names=X_test_fs.columns,
            plot_type="bar", show=False)
        plt.title(f"SHAP Mean |Impact| - {best_model_name}")
        plt.tight_layout()
        bar_path = PLOTS_DIR / "shap_bar.png"
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved plot:", bar_path)
        return path
    except Exception as e:
        print("SHAP could not be generated for this model/environment.")
        print("Reason:", e)
        print("\nFeature importance from the previous cells can still be used.")
        return None


def main():
    config.ensure_directories()
    X_train_fs, _, X_test_fs, _ = load_feature_store()
    leaderboard = pd.read_csv(LEADERBOARD_PATH)
    best_model_name, _, _, winning_model = identify_best_model(leaderboard)
    generate_shap_summary(winning_model, best_model_name, X_train_fs, X_test_fs)


if __name__ == "__main__":
    main()
