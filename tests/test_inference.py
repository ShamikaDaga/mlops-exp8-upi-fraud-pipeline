"""Model-inference tests using the exported serving artefacts."""
import pytest

import src.config  # noqa: F401
from src.inference import FraudPredictor, ValidationError
from tests.test_validation import VALID

FRAUDULENT = dict(
    VALID, Account_Age_Days=9, User_Age=22, User_City="Delhi", KYC_Status=1,
    Transaction_Amount=48000.0, Transaction_Hour=2, Transaction_Type="Merchant",
    Merchant_Category="E-Commerce", Device_Type="Web Browser", Device_Change_Flag=1,
    App_Version="v4.8.5", Previous_Transaction_Count=3, Avg_Transaction_Value=1200.0,
    Transactions_Last_24h=9, Login_Attempts=5, IP_Location_Match=0,
    Suspicious_Device=1, Rapid_Transaction_Flag=1, Wallet_Balance=2500.0,
    Bank_Balance=52000.0,
)


@pytest.fixture(scope="module")
def predictor():
    return FraudPredictor()


def test_prediction_output_contract(predictor):
    result = predictor.predict(VALID)
    assert set(result) == {"fraud_probability", "threshold", "prediction", "is_fraud"}
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["prediction"] in {"Fraud", "Legitimate"}


def test_normal_transaction_is_legitimate(predictor):
    assert predictor.predict(VALID)["prediction"] == "Legitimate"


def test_risky_transaction_is_flagged(predictor):
    assert predictor.predict(FRAUDULENT)["prediction"] == "Fraud"


def test_invalid_input_raises(predictor):
    with pytest.raises(ValidationError):
        predictor.predict(dict(VALID, Transaction_Amount=-1))
