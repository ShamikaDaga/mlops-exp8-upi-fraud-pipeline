"""Exploratory data analysis plots (saved to artifacts/plots/eda_*.png).

Run from the project root:   python -m eda.eda_plots
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402
from src.config import PLOTS_DIR  # noqa: E402


def save_figure(filename):
    """Save the current matplotlib figure into artifacts/plots and close it."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot:", path)
    return path


def plot_fraud_distribution(df):
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Fraud_Flag")
    plt.title("Fraud vs Legitimate Transactions")
    plt.xlabel("Fraud Flag (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Transaction Count")
    return save_figure("eda_01_fraud_distribution.png")


def plot_fraud_by_transaction_type(df):
    fraud_by_type = (
        df.groupby("Transaction_Type")["Fraud_Flag"]
        .agg(["count", "mean"]).sort_values("mean", ascending=False)
    )
    fraud_by_type["Fraud_Rate_%"] = (fraud_by_type["mean"] * 100).round(2)
    print(fraud_by_type.to_string())
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="Transaction_Type", y="Fraud_Flag")
    plt.title("Fraud Rate by Transaction Type")
    plt.ylabel("Fraud Rate")
    plt.xlabel("Transaction Type")
    plt.xticks(rotation=30)
    return save_figure("eda_02_fraud_by_transaction_type.png")


def plot_amount_vs_fraud(df):
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="Fraud_Flag", y="Transaction_Amount")
    plt.title("Transaction Amount vs Fraud")
    plt.xlabel("Fraud Flag")
    plt.ylabel("Transaction Amount (₹)")
    return save_figure("eda_03_transaction_amount_vs_fraud.png")


def plot_fraud_by_hour(df):
    hourly_fraud = df.groupby("Transaction_Hour")["Fraud_Flag"].agg(["count", "mean"])
    hourly_fraud["Fraud_Rate_%"] = hourly_fraud["mean"] * 100
    print(hourly_fraud.to_string())
    plt.figure(figsize=(12, 5))
    sns.barplot(x=hourly_fraud.index, y=hourly_fraud["Fraud_Rate_%"])
    plt.title("Fraud Rate by Transaction Hour")
    plt.xlabel("Transaction Hour")
    plt.ylabel("Fraud Rate (%)")
    return save_figure("eda_04_fraud_rate_by_hour.png")


def plot_device_and_kyc(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(data=df, x="Device_Type", y="Fraud_Flag", ax=axes[0])
    axes[0].set_title("Fraud Rate by Device Type")
    axes[0].set_ylabel("Fraud Rate")
    axes[0].tick_params(axis="x", rotation=30)
    sns.barplot(data=df, x="KYC_Status", y="Fraud_Flag", ax=axes[1])
    axes[1].set_title("Fraud Rate by KYC Status")
    axes[1].set_ylabel("Fraud Rate")
    plt.tight_layout()
    return save_figure("eda_05_device_and_kyc.png")


def plot_behavioral_heatmap(df):
    behavioral_heatmap = df.pivot_table(
        values="Fraud_Flag", index="Login_Attempts",
        columns="Transactions_Last_24h", aggfunc="mean",
    )
    plt.figure(figsize=(12, 7))
    sns.heatmap(behavioral_heatmap, annot=True, fmt=".2f", cmap="YlOrRd")
    plt.title("Fraud Rate: Login Attempts vs Transactions in Last 24h")
    plt.xlabel("Transactions Last 24h")
    plt.ylabel("Login Attempts")
    return save_figure("eda_06_behavioral_risk_heatmap.png")


def plot_correlation_heatmap(train_engineered):
    numeric_for_corr = train_engineered.select_dtypes(include=np.number)
    corr_matrix = numeric_for_corr.corr()
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Numeric and Engineered Features")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return save_figure("eda_07_correlation_heatmap.png")


def _engineered_training_frame(df):
    """Re-create the engineered *training* frame exactly as the ETL stage does."""
    from src.feature_engineering import calculate_imputation_values, engineer_features
    from src.preprocessing import split_data

    train_df, _ = split_data(df)
    imputation = calculate_imputation_values(train_df)
    return engineer_features(train_df, train_df, imputation)


def run_eda():
    """Generate every EDA visual from the notebook."""
    from src.data_loader import (
        convert_data_types, encode_target_and_binary, load_raw_dataframe,
    )

    config.ensure_directories()
    df = load_raw_dataframe()
    df = convert_data_types(df)
    df = encode_target_and_binary(df)

    plot_fraud_distribution(df)
    plot_fraud_by_transaction_type(df)
    plot_amount_vs_fraud(df)
    plot_fraud_by_hour(df)
    plot_device_and_kyc(df)
    plot_behavioral_heatmap(df)
    # The correlation heatmap uses the engineered training data.
    plot_correlation_heatmap(_engineered_training_frame(df))


if __name__ == "__main__":
    run_eda()
