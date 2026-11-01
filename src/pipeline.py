"""ETL stage: raw Excel -> cleaned -> engineered -> feature store (+ MLflow run).

Run from the project root:   python -m src.pipeline
"""
import json

import joblib

from src import config
from src.config import (
    FEATURE_NAMES_PATH, FILE_PATH, PREPROCESSOR_PATH, RANDOM_STATE, SERVING_DIR,
    TEST_FEATURE_STORE_PATH, TEST_SPLIT_PATH, TRAIN_FEATURE_STORE_PATH,
)
from src.data_loader import (
    amount_outlier_fraud_analysis, convert_data_types, encode_target_and_binary,
    initial_inspection, inspect_structure, load_raw_dataframe,
    missing_value_summary, outlier_summary_iqr, validate_data,
)
from src.feature_engineering import (
    calculate_imputation_values, engineer_features, verify_engineered_features,
    verify_missing_values,
)
from src.preprocessing import (
    build_preprocessor, create_feature_store, fit_transform_preprocessor,
    identify_feature_types, load_feature_store, prepare_xy, save_clean_engineered,
    save_preprocessor, split_data, verify_feature_store_reuse,
    verify_final_model_data,
)

CATEGORICAL_SERVING_COLUMNS = [
    "User_City", "Transaction_Type", "Merchant_Category", "Device_Type", "App_Version",
]


def _write_serving_metadata(train_df):
    """Persist the lookups the Streamlit service needs (dropdown values, ranges)."""
    meta = {"categories": {}, "ranges": {}}
    for col in CATEGORICAL_SERVING_COLUMNS:
        meta["categories"][col] = sorted(train_df[col].dropna().astype(str).unique().tolist())
    for col in [
        "Account_Age_Days", "User_Age", "Transaction_Amount", "Previous_Transaction_Count",
        "Avg_Transaction_Value", "Transactions_Last_24h", "Login_Attempts",
        "Wallet_Balance", "Bank_Balance",
    ]:
        meta["ranges"][col] = {
            "min": float(train_df[col].min()),
            "max": float(train_df[col].max()),
            "median": float(train_df[col].median()),
        }
    with open(SERVING_DIR / "serving_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def run_etl():
    config.ensure_directories()
    mlflow = config.setup_mlflow()

    print("\n=== EXTRACT ===")
    df = load_raw_dataframe()
    print(df.head().to_string(max_cols=8))

    print("\n=== INSPECT ===")
    inspect_structure(df)
    initial_inspection(df)
    print("\n=== MISSING VALUES ===")
    missing_value_summary(df)
    print("\n=== DATA TYPES ===")
    df = convert_data_types(df)
    print("\n=== VALIDATION ===")
    validate_data(df)
    print("\n=== TARGET / BINARY ENCODING ===")
    df = encode_target_and_binary(df)
    print("\n=== OUTLIERS (IQR) ===")
    outlier_summary_iqr(df)
    amount_outlier_fraud_analysis(df)

    print("\n=== TRAIN / TEST SPLIT ===")
    train_df, test_df = split_data(df)
    joblib.dump(test_df, TEST_SPLIT_PATH)

    print("\n=== IMPUTATION VALUES ===")
    imputation = calculate_imputation_values(train_df)

    print("\n=== FEATURE ENGINEERING ===")
    train_engineered = engineer_features(train_df, train_df, imputation)
    test_engineered = engineer_features(test_df, train_df, imputation)
    print("Training engineered shape:")
    print(train_engineered.shape)
    print("\nTesting engineered shape:")
    print(test_engineered.shape)
    print(train_engineered.head().to_string(max_cols=8))
    verify_missing_values(train_engineered, test_engineered)
    verify_engineered_features()

    print("\n=== X / y + PREPROCESSOR ===")
    X_train_raw, X_test_raw, y_train, y_test = prepare_xy(train_engineered, test_engineered)
    numeric_features, categorical_features = identify_feature_types(X_train_raw)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_processed, X_test_processed, feature_names = fit_transform_preprocessor(
        preprocessor, X_train_raw, X_test_raw, y_train
    )
    print(X_train_processed.head().to_string(max_cols=8))
    verify_final_model_data(X_train_processed, X_test_processed, y_train, y_test)
    save_preprocessor(preprocessor)

    print("\n=== FEATURE STORE ===")
    create_feature_store(X_train_processed, y_train, X_test_processed, y_test, feature_names)
    X_train_fs, y_train_fs, X_test_fs, y_test_fs = load_feature_store()
    verify_feature_store_reuse(
        X_train_fs, X_test_fs, y_train_fs, y_test_fs,
        X_train_processed, X_test_processed, y_train, y_test,
    )
    save_clean_engineered(train_engineered, test_engineered)

    # Serving artefacts: everything the inference service needs at runtime.
    joblib.dump(train_df, SERVING_DIR / "reference_train.joblib")
    joblib.dump(imputation, SERVING_DIR / "imputation_values.joblib")
    joblib.dump(preprocessor, SERVING_DIR / "preprocessor.joblib")
    _write_serving_metadata(train_df)

    print("\n=== MLFLOW: FEATURE STORE BUILD ===")
    with mlflow.start_run(run_name="FeatureStore_Build"):
        mlflow.set_tag("stage", "ETL_and_Feature_Store")
        mlflow.log_param("dataset", FILE_PATH.name)
        mlflow.log_param("dataset_rows", len(df))
        mlflow.log_param("dataset_columns", len(df.columns))
        mlflow.log_param("train_test_split", "80/20")
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("feature_count", len(feature_names))
        mlflow.log_artifact(str(TRAIN_FEATURE_STORE_PATH), artifact_path="feature_store")
        mlflow.log_artifact(str(TEST_FEATURE_STORE_PATH), artifact_path="feature_store")
        mlflow.log_artifact(str(FEATURE_NAMES_PATH), artifact_path="feature_store")
        mlflow.log_artifact(str(PREPROCESSOR_PATH), artifact_path="feature_store")
    print("Feature-store build successfully tracked in MLflow.")
    return df, train_df, test_df


if __name__ == "__main__":
    run_etl()
