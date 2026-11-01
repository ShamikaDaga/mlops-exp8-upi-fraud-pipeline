"""Split, preprocessing transformer and feature-store helpers."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    COLUMNS_TO_DROP, FEATURE_NAMES_PATH, PREPROCESSOR_PATH, RANDOM_STATE,
    TARGET, TEST_FEATURE_STORE_PATH, TEST_SIZE, TRAIN_FEATURE_STORE_PATH,
    CLEAN_ENGINEERED_PATH,
)


def split_data(df):
    """Stratified 80/20 train-test split."""
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET]
    )
    train_df = train_df.copy()
    test_df = test_df.copy()
    print("Training shape:", train_df.shape)
    print("Testing shape:", test_df.shape)
    print("\nTraining fraud rate:")
    print(train_df[TARGET].mean() * 100)
    print("\nTesting fraud rate:")
    print(test_df[TARGET].mean() * 100)
    return train_df, test_df


def prepare_xy(train_engineered, test_engineered):
    """Drop identifier / leakage columns and split features from target."""
    X_train_raw = train_engineered.drop(columns=COLUMNS_TO_DROP)
    X_test_raw = test_engineered.drop(columns=COLUMNS_TO_DROP)
    y_train = train_engineered[TARGET].astype(int)
    y_test = test_engineered[TARGET].astype(int)
    print("X_train shape:", X_train_raw.shape)
    print("X_test shape:", X_test_raw.shape)
    print("\ny_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    return X_train_raw, X_test_raw, y_train, y_test


def identify_feature_types(X_train_raw):
    """Separate categorical and numerical feature names."""
    categorical_features = X_train_raw.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()
    numeric_features = [c for c in X_train_raw.columns if c not in categorical_features]
    print("Numerical features:")
    print(numeric_features)
    print("\nCategorical features:")
    print(categorical_features)
    print("\nNumber of numerical features:", len(numeric_features))
    print("Number of categorical features:", len(categorical_features))
    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    """ColumnTransformer: median+scale for numerics, mode+one-hot for categoricals."""
    # Compatibility with different scikit-learn versions
    try:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", one_hot_encoder),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    print("Preprocessor created successfully.")
    return preprocessor


def fit_transform_preprocessor(preprocessor, X_train_raw, X_test_raw, y_train):
    """Fit on training data only, transform both, return DataFrames."""
    preprocessor.fit(X_train_raw, y_train)
    X_train_processed = preprocessor.transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out()
    print("Processed training shape:")
    print(X_train_processed.shape)
    print("\nProcessed testing shape:")
    print(X_test_processed.shape)
    print("\nTotal final model features:")
    print(len(feature_names))

    X_train_processed = pd.DataFrame(
        X_train_processed, columns=feature_names, index=X_train_raw.index
    )
    X_test_processed = pd.DataFrame(
        X_test_processed, columns=feature_names, index=X_test_raw.index
    )
    return X_train_processed, X_test_processed, feature_names


def verify_final_model_data(X_train_processed, X_test_processed, y_train, y_test):
    print("Any missing values in processed training data?", X_train_processed.isna().any().any())
    print("Any missing values in processed testing data?", X_test_processed.isna().any().any())
    print("\nTraining class distribution:")
    print(y_train.value_counts())
    print("\nTesting class distribution:")
    print(y_test.value_counts())


def save_preprocessor(preprocessor):
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print("Preprocessor saved at:", PREPROCESSOR_PATH)


def create_feature_store(X_train_processed, y_train, X_test_processed, y_test, feature_names):
    """Persist the processed train/test features as the feature store."""
    train_feature_store = X_train_processed.copy()
    train_feature_store[TARGET] = y_train
    test_feature_store = X_test_processed.copy()
    test_feature_store[TARGET] = y_test
    train_feature_store.to_csv(TRAIN_FEATURE_STORE_PATH, index=False)
    test_feature_store.to_csv(TEST_FEATURE_STORE_PATH, index=False)
    # Save feature names
    pd.DataFrame({"Feature_Name": feature_names}).to_csv(FEATURE_NAMES_PATH, index=False)
    print("Feature store created successfully.")
    print("\nTraining features:")
    print(TRAIN_FEATURE_STORE_PATH)
    print("\nTesting features:")
    print(TEST_FEATURE_STORE_PATH)
    print("\nFeature names:")
    print(FEATURE_NAMES_PATH)


def load_feature_store():
    """Reload the stored features: X_train, y_train, X_test, y_test."""
    stored_train = pd.read_csv(TRAIN_FEATURE_STORE_PATH)
    stored_test = pd.read_csv(TEST_FEATURE_STORE_PATH)
    X_train_fs = stored_train.drop(columns=[TARGET])
    X_test_fs = stored_test.drop(columns=[TARGET])
    y_train_fs = stored_train[TARGET].astype(int)
    y_test_fs = stored_test[TARGET].astype(int)
    return X_train_fs, y_train_fs, X_test_fs, y_test_fs


def verify_feature_store_reuse(X_train_fs, X_test_fs, y_train_fs, y_test_fs,
                               X_train_processed, X_test_processed, y_train, y_test):
    print("Feature store successfully reloaded.")
    print("\nTraining feature shape:", X_train_fs.shape)
    print("Testing feature shape:", X_test_fs.shape)
    print("Training feature store matches original:",
          np.allclose(X_train_fs.values, X_train_processed.values))
    print("Testing feature store matches original:",
          np.allclose(X_test_fs.values, X_test_processed.values))
    print("Training target matches:", np.array_equal(y_train_fs.values, y_train.values))
    print("Testing target matches:", np.array_equal(y_test_fs.values, y_test.values))


def save_clean_engineered(train_engineered, test_engineered):
    """Save the complete clean engineered dataset."""
    clean_engineered = pd.concat([train_engineered, test_engineered]).sort_index()
    clean_engineered.to_csv(CLEAN_ENGINEERED_PATH, index=False)
    print("Clean engineered dataset saved at:")
    print(CLEAN_ENGINEERED_PATH)
