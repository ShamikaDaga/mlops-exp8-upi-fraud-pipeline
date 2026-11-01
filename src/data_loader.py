"""Extract + clean stage: load the Excel file, inspect, convert, validate."""
import pandas as pd

from src.config import FILE_PATH, SHEET_NAME

NUMERIC_COLUMNS = [
    "Account_Age_Days", "User_Age", "Transaction_Amount", "Transaction_Hour",
    "Device_Change_Flag", "Previous_Transaction_Count", "Avg_Transaction_Value",
    "Transactions_Last_24h", "Login_Attempts", "Wallet_Balance", "Bank_Balance",
]
OUTLIER_COLUMNS = ["Transaction_Amount", "Avg_Transaction_Value", "Login_Attempts"]


def load_raw_dataframe(file_path=FILE_PATH, sheet_name=SHEET_NAME):
    """Extract: load the Excel sheet and promote row 3 to column names."""
    raw_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    print("Raw Excel shape:", raw_df.shape)
    # Row index 3 contains the actual column names
    column_names = (
        raw_df.iloc[3].astype(str).str.strip().str.replace(" ", "_", regex=False)
    )
    # Data starts from row index 4
    df = raw_df.iloc[4:].copy()
    df.columns = column_names
    df = df.reset_index(drop=True)
    # Remove any accidental column-name metadata
    df.columns.name = None
    print("Clean extracted shape:", df.shape)
    return df


def inspect_structure(df):
    """Print rows, columns and the numbered column list."""
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])
    print("\nColumn names:")
    for i, col in enumerate(df.columns, start=1):
        print(i, ":", col)


def initial_inspection(df):
    """Print head, tail and dtype information."""
    print("First 5 rows:")
    print(df.head().to_string(max_cols=8))
    print("\nLast 5 rows:")
    print(df.tail().to_string(max_cols=8))
    print("\nDataset information:")
    df.info()


def missing_value_summary(df):
    """Count / percentage of missing values per column."""
    summary = pd.DataFrame({
        "Missing_Count": df.isna().sum(),
        "Total_Rows": len(df),
        "Missing_Percentage": (df.isna().sum() / len(df) * 100).round(2),
    })
    summary = summary.sort_values("Missing_Count", ascending=False)
    print(summary.to_string())
    return summary


def convert_data_types(df):
    """Numeric / date / time conversions."""
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Date
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    # Time as string
    df["Transaction_Time"] = df["Transaction_Time"].astype("string").str.strip()
    print("Data types after conversion:")
    print(df.dtypes.to_string())
    return df


def validate_data(df):
    """Data-quality checks (duplicates, impossible values, hour mismatches)."""
    # Parse hour from Transaction_Time
    parsed_hour = pd.to_datetime(
        df["Transaction_Time"], format="%H:%M", errors="coerce"
    ).dt.hour
    print("Duplicate complete rows:", df.duplicated().sum())
    print("Duplicate Transaction IDs:", df["Transaction_ID"].duplicated().sum())
    print("Invalid Transaction Amounts:", (df["Transaction_Amount"] <= 0).sum())
    print(
        "Invalid Account Ages:",
        ((df["Account_Age_Days"] < 0) | (df["Account_Age_Days"] > 1825)).sum(),
    )
    print(
        "Invalid User Ages:",
        ((df["User_Age"] < 18) | (df["User_Age"] > 64)).sum(),
    )
    print(
        "Invalid Transaction Hours:",
        ((df["Transaction_Hour"] < 0) | (df["Transaction_Hour"] > 23)).sum(),
    )
    print(
        "Transaction hour mismatches:",
        (parsed_hour.notna() & (parsed_hour != df["Transaction_Hour"])).sum(),
    )
    print("\nTarget values:")
    print(df["Fraud_Flag"].value_counts(dropna=False))


def encode_target_and_binary(df):
    """Convert the target, binary security columns and the ordinal KYC column."""
    # Target
    df["Fraud_Flag"] = df["Fraud_Flag"].map({"No": 0, "Yes": 1})
    # Binary security variables
    binary_mapping = {"No": 0, "Yes": 1}
    for col in ["IP_Location_Match", "Suspicious_Device", "Rapid_Transaction_Flag"]:
        df[col] = df[col].map(binary_mapping)
    # Device change is already numeric
    df["Device_Change_Flag"] = pd.to_numeric(df["Device_Change_Flag"], errors="coerce")
    # KYC is ordinal
    df["KYC_Status"] = df["KYC_Status"].map(
        {"Not Submitted": 0, "Pending": 1, "Verified": 2}
    )
    print("Target distribution:")
    print(df["Fraud_Flag"].value_counts())
    print("\nKYC encoding:")
    print(df["KYC_Status"].value_counts())
    return df


def outlier_summary_iqr(df, columns=OUTLIER_COLUMNS):
    """IQR based outlier summary for the chosen columns."""
    outlier_results = []
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_results.append({
            "Column": col, "Q1": q1, "Q3": q3, "IQR": iqr,
            "Lower_Bound": lower_bound, "Upper_Bound": upper_bound,
            "Outlier_Count": outlier_mask.sum(),
            "Outlier_Percentage": outlier_mask.mean() * 100,
        })
    summary = pd.DataFrame(outlier_results)
    print(summary.to_string())
    return summary


def amount_outlier_fraud_analysis(df):
    """Are transaction-amount outliers fraud signals?"""
    q1 = df["Transaction_Amount"].quantile(0.25)
    q3 = df["Transaction_Amount"].quantile(0.75)
    iqr = q3 - q1
    amount_upper = q3 + 1.5 * iqr
    amount_outlier = df["Transaction_Amount"] > amount_upper
    print("Transaction amount IQR upper bound:")
    print(f"₹ {amount_upper:,.2f}")
    print("\nFraud rate among amount outliers:")
    print(df.loc[amount_outlier, "Fraud_Flag"].mean() * 100, "%")
    print("\nFraud rate among non-outliers:")
    print(df.loc[~amount_outlier, "Fraud_Flag"].mean() * 100, "%")
