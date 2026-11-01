"""Model definitions for both imbalance-handling methods."""
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

METHODS = ["Class Weights", "SMOTE"]

# Model parameter dictionary logged to MLflow
MODEL_PARAMETERS = {
    "Logistic Regression": {"max_iter": 1000},
    "Decision Tree": {"max_depth": 8, "min_samples_leaf": 5},
    "Random Forest": {"n_estimators": 300, "min_samples_leaf": 2},
    "XGBoost": {
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
    },
    "LightGBM": {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05},
    "CatBoost": {"iterations": 300, "depth": 6, "learning_rate": 0.05},
    "AdaBoost": {"n_estimators": 300, "learning_rate": 0.05},
}


def create_models(method, scale_pos_weight):
    """Return the seven classifiers configured for the given imbalance method."""
    if method == "Class Weights":
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
                random_state=42, eval_metric="logloss", n_jobs=1),
            "LightGBM": LGBMClassifier(
                n_estimators=300, num_leaves=31, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, class_weight="balanced", random_state=42,
                verbosity=-1, n_jobs=1),
            "CatBoost": CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.05,
                auto_class_weights="Balanced", random_seed=42, verbose=0, thread_count=1),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=300, learning_rate=0.05, random_state=42),
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=5, random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=42, eval_metric="logloss", n_jobs=1),
            "LightGBM": LGBMClassifier(
                n_estimators=300, num_leaves=31, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=1),
            "CatBoost": CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.05, random_seed=42,
                verbose=0, thread_count=1),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=300, learning_rate=0.05, random_state=42),
        }
    return models
