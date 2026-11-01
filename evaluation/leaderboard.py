"""Model leaderboard: ranking, saving and best-model selection."""
import joblib
import pandas as pd

from src.config import LEADERBOARD_PATH, TRAINING_RESULTS_PATH
from models.train_models import model_artifact_path


def build_leaderboard(results=None):
    """Rank all runs by F1-Macro then ROC-AUC."""
    if results is None:
        results = pd.read_csv(TRAINING_RESULTS_PATH)
    leaderboard = pd.DataFrame(results)
    leaderboard = leaderboard.sort_values(
        by=["F1-Macro", "ROC-AUC"], ascending=False
    ).reset_index(drop=True)
    print("=== MODEL LEADERBOARD ===")
    print(leaderboard.to_string(float_format=lambda x: f"{x:.4f}"))
    return leaderboard


def save_leaderboard(leaderboard):
    leaderboard.to_csv(LEADERBOARD_PATH, index=False)
    print("Leaderboard saved at:", LEADERBOARD_PATH)


def identify_best_model(leaderboard):
    """Return (name, method, row, fitted model) of the top-ranked run."""
    best_row = leaderboard.iloc[0]
    best_model_name = best_row["Model"]
    best_method = best_row["Method"]
    winning_model = joblib.load(model_artifact_path(best_model_name, best_method))
    print("=== BEST MODEL ===")
    print("Model:", best_model_name)
    print("Training Method:", best_method)
    print("F1-Macro:", round(best_row["F1-Macro"], 4))
    print("F1-Fraud:", round(best_row["F1-Fraud"], 4))
    print("ROC-AUC:", round(best_row["ROC-AUC"], 4))
    print("PR-AUC:", round(best_row["PR-AUC"], 4))
    return best_model_name, best_method, best_row, winning_model
