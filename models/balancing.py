"""Class-imbalance handling: SMOTE oversampling and class / scale weights."""
from imblearn.over_sampling import SMOTE

from src.config import RANDOM_STATE


def apply_smote(X_train_fs, y_train_fs):
    """Handle class imbalance using SMOTE (training data only)."""
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_fs, y_train_fs)
    print("Before SMOTE:")
    print(y_train_fs.value_counts())
    print("\nAfter SMOTE:")
    print(y_train_smote.value_counts())
    print("\nOriginal training rows:", len(X_train_fs))
    print("SMOTE training rows:", len(X_train_smote))
    return X_train_smote, y_train_smote


def calculate_scale_pos_weight(y_train_fs):
    """Class weight / XGBoost scale_pos_weight = negatives / positives."""
    negative_count = (y_train_fs == 0).sum()
    positive_count = (y_train_fs == 1).sum()
    scale_pos_weight = negative_count / positive_count
    print("Legitimate transactions:", negative_count)
    print("Fraud transactions:", positive_count)
    print("XGBoost scale_pos_weight:", round(scale_pos_weight, 3))
    return scale_pos_weight
