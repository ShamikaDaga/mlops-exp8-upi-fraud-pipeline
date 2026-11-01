"""Data-loading and feature-engineering tests (use the raw Excel file)."""
import pytest

import src.config  # noqa: F401  (applies thread settings)
from src.config import FILE_PATH
from src.data_loader import convert_data_types, encode_target_and_binary, load_raw_dataframe
from src.feature_engineering import (
    ENGINEERED_FEATURES, calculate_imputation_values, engineer_features,
)
from src.preprocessing import split_data


@pytest.fixture(scope="module")
def engineered():
    assert FILE_PATH.exists(), "raw dataset missing"
    df = encode_target_and_binary(convert_data_types(load_raw_dataframe()))
    train_df, test_df = split_data(df)
    imputation = calculate_imputation_values(train_df)
    return (
        df, engineer_features(train_df, train_df, imputation),
        engineer_features(test_df, train_df, imputation),
    )


def test_dataset_shape(engineered):
    df, _, _ = engineered
    assert df.shape == (2800, 26)
    assert set(df["Fraud_Flag"].unique()) == {0, 1}


def test_split_is_80_20_stratified(engineered):
    df, train, test = engineered
    assert len(train) == 2240 and len(test) == 560
    assert abs(train["Fraud_Flag"].mean() - test["Fraud_Flag"].mean()) < 0.01


def test_all_engineered_features_created(engineered):
    _, train, _ = engineered
    missing = [f for f in ENGINEERED_FEATURES if f not in train.columns]
    assert not missing


def test_no_missing_values_after_cleaning(engineered):
    _, train, test = engineered
    for frame in (train, test):
        assert frame.drop(columns=["Merchant_Category"]).isna().sum().sum() == 0
        assert frame["Merchant_Category"].isna().sum() == 0
