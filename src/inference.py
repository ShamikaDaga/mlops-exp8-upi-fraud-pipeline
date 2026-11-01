"""Inference helpers shared by the Streamlit app, the tests and the notebook-style scripts."""
import json
from datetime import date, datetime

import joblib
import pandas as pd

from src.config import COLUMNS_TO_DROP, SERVING_DIR
from src.feature_engineering import engineer_features

KYC_LABELS = {"Not Submitted": 0, "Pending": 1, "Verified": 2}

# Field name -> (min, max) accepted by the service.
VALIDATION_RULES = {
    "Account_Age_Days": (0, 1825),
    "User_Age": (18, 64),
    "Transaction_Hour": (0, 23),
    "Login_Attempts": (0, 20),
    "Transactions_Last_24h": (0, 100),
    "Previous_Transaction_Count": (0, 100000),
}
REQUIRED_FIELDS = [
    "Account_Age_Days", "User_Age", "User_City", "KYC_Status", "Transaction_Amount",
    "Transaction_Date", "Transaction_Hour", "Transaction_Type", "Merchant_Category",
    "Device_Type", "Device_Change_Flag", "App_Version", "Previous_Transaction_Count",
    "Avg_Transaction_Value", "Transactions_Last_24h", "Login_Attempts",
    "IP_Location_Match", "Suspicious_Device", "Rapid_Transaction_Flag",
    "Wallet_Balance", "Bank_Balance",
]


class ValidationError(ValueError):
    """Raised when a prediction request contains invalid input."""


def validate_inputs(inputs):
    """Raise ValidationError with a readable message for any invalid field."""
    missing = [f for f in REQUIRED_FIELDS if inputs.get(f) is None]
    if missing:
        raise ValidationError("Missing required fields: " + ", ".join(missing))
    if float(inputs["Transaction_Amount"]) <= 0:
        raise ValidationError("Transaction amount must be greater than 0")
    for field, (low, high) in VALIDATION_RULES.items():
        value = float(inputs[field])
        if not low <= value <= high:
            raise ValidationError(f"{field} must be between {low} and {high}")
    for field in ("Avg_Transaction_Value", "Wallet_Balance", "Bank_Balance"):
        if float(inputs[field]) < 0:
            raise ValidationError(f"{field} cannot be negative")
    for field in ("Device_Change_Flag", "IP_Location_Match",
                  "Suspicious_Device", "Rapid_Transaction_Flag"):
        if int(inputs[field]) not in (0, 1):
            raise ValidationError(f"{field} must be 0 or 1")
    if int(inputs["KYC_Status"]) not in (0, 1, 2):
        raise ValidationError("KYC_Status must be 0, 1 or 2")


class FraudPredictor:
    """Loads the exported model + preprocessing state once and scores transactions."""

    def __init__(self, serving_dir=SERVING_DIR):
        self.model = joblib.load(serving_dir / "best_model.joblib")
        self.preprocessor = joblib.load(serving_dir / "preprocessor.joblib")
        self.reference_train = joblib.load(serving_dir / "reference_train.joblib")
        self.imputation = joblib.load(serving_dir / "imputation_values.joblib")
        with open(serving_dir / "model_meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(serving_dir / "serving_metadata.json", encoding="utf-8") as f:
            self.options = json.load(f)
        self.threshold = float(self.meta["optimal_threshold"])
        self.model_name = self.meta["best_model"]

    @staticmethod
    def build_raw_frame(inputs):
        """Turn the user's inputs into a one-row frame shaped like the raw data."""
        txn_date = inputs["Transaction_Date"]
        if isinstance(txn_date, str):
            txn_date = datetime.strptime(txn_date, "%Y-%m-%d")
        elif isinstance(txn_date, date) and not isinstance(txn_date, datetime):
            txn_date = datetime(txn_date.year, txn_date.month, txn_date.day)
        hour = int(inputs["Transaction_Hour"])
        merchant_category = (
            inputs["Merchant_Category"] if inputs["Transaction_Type"] == "Merchant" else None
        )
        row = {
            "User_ID": "LIVE_USER", "Account_Age_Days": float(inputs["Account_Age_Days"]),
            "User_Age": float(inputs["User_Age"]), "User_City": inputs["User_City"],
            "KYC_Status": int(inputs["KYC_Status"]), "Transaction_ID": "LIVE_TXN",
            "Transaction_Amount": float(inputs["Transaction_Amount"]),
            "Transaction_Date": pd.Timestamp(txn_date),
            "Transaction_Time": f"{hour:02d}:00", "Transaction_Hour": float(hour),
            "Transaction_Type": inputs["Transaction_Type"],
            "Merchant_Category": merchant_category,
            "Device_Type": inputs["Device_Type"],
            "Device_Change_Flag": float(inputs["Device_Change_Flag"]),
            "Operating_System": "NA", "App_Version": inputs["App_Version"],
            "Previous_Transaction_Count": float(inputs["Previous_Transaction_Count"]),
            "Avg_Transaction_Value": float(inputs["Avg_Transaction_Value"]),
            "Transactions_Last_24h": float(inputs["Transactions_Last_24h"]),
            "Login_Attempts": float(inputs["Login_Attempts"]),
            "IP_Location_Match": int(inputs["IP_Location_Match"]),
            "Suspicious_Device": int(inputs["Suspicious_Device"]),
            "Rapid_Transaction_Flag": int(inputs["Rapid_Transaction_Flag"]),
            "Wallet_Balance": float(inputs["Wallet_Balance"]),
            "Bank_Balance": float(inputs["Bank_Balance"]),
            "Fraud_Flag": 0,
        }
        return pd.DataFrame([row])

    def predict(self, inputs):
        """Validate, engineer features, transform and score one transaction."""
        validate_inputs(inputs)
        raw = self.build_raw_frame(inputs)
        engineered = engineer_features(raw, self.reference_train, self.imputation)
        features = engineered.drop(columns=COLUMNS_TO_DROP)
        processed = pd.DataFrame(
            self.preprocessor.transform(features),
            columns=self.preprocessor.get_feature_names_out(),
        )
        probability = float(self.model.predict_proba(processed)[0, 1])
        is_fraud = probability >= self.threshold
        return {
            "fraud_probability": probability,
            "threshold": self.threshold,
            "prediction": "Fraud" if is_fraud else "Legitimate",
            "is_fraud": bool(is_fraud),
        }
