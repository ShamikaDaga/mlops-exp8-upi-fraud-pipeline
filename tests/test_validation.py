"""Input-validation tests for the prediction service."""
import pytest

from src.inference import ValidationError, validate_inputs

VALID = dict(
    Account_Age_Days=1200, User_Age=34, User_City="Mumbai", KYC_Status=2,
    Transaction_Amount=850.0, Transaction_Date="2024-10-05", Transaction_Hour=14,
    Transaction_Type="P2P", Merchant_Category="Grocery", Device_Type="Android",
    Device_Change_Flag=0, App_Version="v5.3.0", Previous_Transaction_Count=420,
    Avg_Transaction_Value=900.0, Transactions_Last_24h=2, Login_Attempts=1,
    IP_Location_Match=1, Suspicious_Device=0, Rapid_Transaction_Flag=0,
    Wallet_Balance=18000.0, Bank_Balance=240000.0,
)


def test_valid_request_passes():
    validate_inputs(VALID)


@pytest.mark.parametrize("field,value,message", [
    ("Transaction_Amount", 0, "greater than 0"),
    ("Transaction_Amount", -5, "greater than 0"),
    ("Transaction_Hour", 25, "Transaction_Hour must be between"),
    ("User_Age", 10, "User_Age must be between"),
    ("Account_Age_Days", -1, "Account_Age_Days must be between"),
    ("Bank_Balance", -100, "cannot be negative"),
    ("IP_Location_Match", 2, "must be 0 or 1"),
    ("KYC_Status", 5, "KYC_Status must be 0, 1 or 2"),
])
def test_invalid_values_are_rejected(field, value, message):
    bad = dict(VALID, **{field: value})
    with pytest.raises(ValidationError, match=message):
        validate_inputs(bad)


def test_missing_field_is_rejected():
    bad = dict(VALID)
    del bad["Login_Attempts"]
    with pytest.raises(ValidationError, match="Missing required fields"):
        validate_inputs(bad)
