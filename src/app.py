"""Streamlit inference service for the UPI fraud model, instrumented with Prometheus.

Run from the project root:   streamlit run src/app.py
Prometheus metrics are exposed on  http://localhost:8000/metrics
"""
import sys
import time
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from prometheus_client import REGISTRY  # noqa: E402

from src import metrics  # noqa: E402
from src.config import PLOTS_DIR, SERVING_DIR  # noqa: E402
from src.inference import (  # noqa: E402
    KYC_LABELS, REQUIRED_FIELDS, FraudPredictor, ValidationError,
)

st.set_page_config(page_title="UPI Fraud Detection | MLOps Exp 8", page_icon="🛡️", layout="wide")

SAMPLES = {
    "Legitimate sample": dict(
        Account_Age_Days=1200, User_Age=34, User_City="Mumbai", KYC="Verified",
        Transaction_Amount=850.0, Transaction_Hour=14, Transaction_Type="P2P",
        Merchant_Category="Grocery", Device_Type="Android", Device_Change=0,
        App_Version="v5.3.0", Previous_Transaction_Count=420, Avg_Transaction_Value=900.0,
        Transactions_Last_24h=2, Login_Attempts=1, IP_Match=1, Suspicious_Device=0,
        Rapid_Txn=0, Wallet_Balance=18000.0, Bank_Balance=240000.0),
    "Fraud-like sample": dict(
        Account_Age_Days=9, User_Age=22, User_City="Delhi", KYC="Pending",
        Transaction_Amount=48000.0, Transaction_Hour=2, Transaction_Type="Merchant",
        Merchant_Category="E-Commerce", Device_Type="Web Browser", Device_Change=1,
        App_Version="v4.8.5", Previous_Transaction_Count=3, Avg_Transaction_Value=1200.0,
        Transactions_Last_24h=9, Login_Attempts=5, IP_Match=0, Suspicious_Device=1,
        Rapid_Txn=1, Wallet_Balance=2500.0, Bank_Balance=52000.0),
}


@st.cache_resource(show_spinner="Loading model and starting metrics endpoint...")
def get_predictor():
    predictor = FraudPredictor()
    metrics.start_metrics_server()
    metrics.APP_UP.set(1)
    metrics.MODEL_LOADED.labels(
        model=predictor.model_name, method=predictor.meta["training_method"]).set(1)
    return predictor


def score(predictor, inputs):
    """Predict + record Prometheus metrics. Returns (result|None, error|None)."""
    start = time.perf_counter()
    try:
        result = predictor.predict(inputs)
    except ValidationError as exc:
        metrics.record_error("validation_error")
        return None, str(exc)
    except Exception as exc:  # unexpected failure while serving
        metrics.record_error(type(exc).__name__)
        return None, f"Unexpected error: {exc}"
    metrics.record_success(result, time.perf_counter() - start)
    return result, None


def to_inputs(form):
    """Map the form values onto the model's raw feature names."""
    return {
        "Account_Age_Days": form["Account_Age_Days"], "User_Age": form["User_Age"],
        "User_City": form["User_City"], "KYC_Status": KYC_LABELS[form["KYC"]],
        "Transaction_Amount": form["Transaction_Amount"],
        "Transaction_Date": form.get("Transaction_Date", date.today()),
        "Transaction_Hour": form["Transaction_Hour"],
        "Transaction_Type": form["Transaction_Type"],
        "Merchant_Category": form["Merchant_Category"],
        "Device_Type": form["Device_Type"], "Device_Change_Flag": form["Device_Change"],
        "App_Version": form["App_Version"],
        "Previous_Transaction_Count": form["Previous_Transaction_Count"],
        "Avg_Transaction_Value": form["Avg_Transaction_Value"],
        "Transactions_Last_24h": form["Transactions_Last_24h"],
        "Login_Attempts": form["Login_Attempts"], "IP_Location_Match": form["IP_Match"],
        "Suspicious_Device": form["Suspicious_Device"],
        "Rapid_Transaction_Flag": form["Rapid_Txn"],
        "Wallet_Balance": form["Wallet_Balance"], "Bank_Balance": form["Bank_Balance"],
    }


def sample_value(key, default):
    return st.session_state.get(key, default)


def load_sample(name):
    for key, value in SAMPLES[name].items():
        st.session_state[f"f_{key}"] = value


def metric_value(name, labels=None):
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


# ------------------------------------------------------------------ layout
predictor = get_predictor()
opts = predictor.options["categories"]

with st.sidebar:
    st.title("🛡️ UPI Fraud Detector")
    st.caption("MLOps Experiment 8 · End-to-End Pipeline")
    st.markdown(f"**Model:** {predictor.model_name}  \n"
                f"**Imbalance method:** {predictor.meta['training_method']}  \n"
                f"**Decision threshold:** {predictor.threshold:.3f}")
    st.divider()
    st.subheader("Traffic simulator")
    st.caption("Replays real transactions from the training data through the "
               "model - every call is counted by Prometheus.")
    n_sim = st.slider("Transactions to simulate", 5, 100, 25, step=5)
    if st.button("▶ Simulate traffic", use_container_width=True):
        sample = predictor.reference_train.sample(n_sim, random_state=int(time.time()))
        counts = {"Fraud": 0, "Legitimate": 0, "errors": 0}
        for _, row in sample.iterrows():
            record = {f: row[f] for f in REQUIRED_FIELDS}
            record["Transaction_Date"] = row["Transaction_Date"].date()
            record["Merchant_Category"] = (
                row["Merchant_Category"] if pd.notna(row["Merchant_Category"]) else "Other")
            for f in REQUIRED_FIELDS:
                if pd.isna(record[f]):
                    record[f] = predictor.imputation["numeric_medians"].get(
                        f, predictor.imputation["categorical_modes"].get(f, 0))
            result, error = score(predictor, record)
            counts["errors" if error else result["prediction"]] += 1
        st.success(f"Simulated {n_sim}: {counts['Fraud']} fraud, "
                   f"{counts['Legitimate']} legitimate, {counts['errors']} errors")
    st.divider()
    st.caption("E024 · SEM 11 · MLOps Experiment 8")

st.title("UPI Fraud Detection - Live Inference Service")
tab_predict, tab_perf, tab_live = st.tabs(
    ["🔍 Predict", "📊 Model performance", "📡 Live service metrics"])

with tab_predict:
    c1, c2, _ = st.columns([1, 1, 4])
    c1.button("Load legitimate sample", on_click=load_sample, args=("Legitimate sample",))
    c2.button("Load fraud-like sample", on_click=load_sample, args=("Fraud-like sample",))
    d = SAMPLES["Legitimate sample"]

    with st.form("txn_form"):
        a, b, c = st.columns(3)
        form = {}
        with a:
            st.markdown("**Transaction**")
            form["Transaction_Amount"] = st.number_input(
                "Transaction amount (₹)", 0.0, 1_000_000.0,
                float(sample_value("f_Transaction_Amount", d["Transaction_Amount"])), 100.0)
            form["Transaction_Hour"] = st.number_input(
                "Transaction hour (0-23)", 0, 23, int(sample_value("f_Transaction_Hour", d["Transaction_Hour"])))
            form["Transaction_Type"] = st.selectbox(
                "Transaction type", opts["Transaction_Type"],
                index=opts["Transaction_Type"].index(sample_value("f_Transaction_Type", d["Transaction_Type"])))
            form["Merchant_Category"] = st.selectbox(
                "Merchant category (merchant payments)", opts["Merchant_Category"],
                index=opts["Merchant_Category"].index(sample_value("f_Merchant_Category", d["Merchant_Category"])))
            form["Transaction_Date"] = st.date_input("Transaction date", date.today())
            form["Transactions_Last_24h"] = st.number_input(
                "Transactions in last 24h", 0, 100, int(sample_value("f_Transactions_Last_24h", d["Transactions_Last_24h"])))
        with b:
            st.markdown("**User & account**")
            form["User_Age"] = st.number_input("User age", 18, 64, int(sample_value("f_User_Age", d["User_Age"])))
            form["User_City"] = st.selectbox(
                "City", opts["User_City"], index=opts["User_City"].index(sample_value("f_User_City", d["User_City"])))
            form["Account_Age_Days"] = st.number_input(
                "Account age (days)", 0, 1825, int(sample_value("f_Account_Age_Days", d["Account_Age_Days"])))
            form["KYC"] = st.selectbox(
                "KYC status", list(KYC_LABELS), index=list(KYC_LABELS).index(sample_value("f_KYC", d["KYC"])))
            form["Previous_Transaction_Count"] = st.number_input(
                "Previous transactions", 0, 100000, int(sample_value("f_Previous_Transaction_Count", d["Previous_Transaction_Count"])))
            form["Avg_Transaction_Value"] = st.number_input(
                "Average transaction value (₹)", 0.0, 1_000_000.0,
                float(sample_value("f_Avg_Transaction_Value", d["Avg_Transaction_Value"])), 100.0)
            form["Wallet_Balance"] = st.number_input(
                "Wallet balance (₹)", 0.0, 10_000_000.0, float(sample_value("f_Wallet_Balance", d["Wallet_Balance"])), 500.0)
            form["Bank_Balance"] = st.number_input(
                "Bank balance (₹)", 0.0, 100_000_000.0, float(sample_value("f_Bank_Balance", d["Bank_Balance"])), 1000.0)
        with c:
            st.markdown("**Device & security**")
            form["Device_Type"] = st.selectbox(
                "Device type", opts["Device_Type"],
                index=opts["Device_Type"].index(sample_value("f_Device_Type", d["Device_Type"])))
            form["App_Version"] = st.selectbox(
                "App version", opts["App_Version"],
                index=opts["App_Version"].index(sample_value("f_App_Version", d["App_Version"])))
            form["Login_Attempts"] = st.number_input(
                "Login attempts", 0, 20, int(sample_value("f_Login_Attempts", d["Login_Attempts"])))
            form["Device_Change"] = int(st.checkbox(
                "Device changed", bool(sample_value("f_Device_Change", d["Device_Change"]))))
            form["IP_Match"] = int(st.checkbox(
                "IP location matches", bool(sample_value("f_IP_Match", d["IP_Match"]))))
            form["Suspicious_Device"] = int(st.checkbox(
                "Suspicious device", bool(sample_value("f_Suspicious_Device", d["Suspicious_Device"]))))
            form["Rapid_Txn"] = int(st.checkbox(
                "Rapid transaction flag", bool(sample_value("f_Rapid_Txn", d["Rapid_Txn"]))))
        submitted = st.form_submit_button("Predict fraud risk", type="primary")

    if submitted:
        result, error = score(predictor, to_inputs(form))
        if error:
            st.error(f"Request rejected: {error}")
        else:
            left, right = st.columns([1, 2])
            left.metric("Fraud probability", f"{result['fraud_probability']:.1%}")
            left.metric("Decision threshold", f"{result['threshold']:.1%}")
            if result["is_fraud"]:
                right.error("🚨 FRAUD - this transaction should be blocked / reviewed")
            else:
                right.success("✅ LEGITIMATE - this transaction looks normal")
            right.progress(min(result["fraud_probability"], 1.0))

with tab_perf:
    st.subheader("Model leaderboard (all models × both balancing methods)")
    st.dataframe(pd.read_csv(SERVING_DIR / "model_leaderboard.csv").round(4), use_container_width=True)
    st.subheader("Final metrics of the deployed model")
    st.dataframe(pd.DataFrame(
        {"Metric": list(predictor.meta["metrics"]),
         "Score": [round(v, 4) for v in predictor.meta["metrics"].values()]}),
        use_container_width=True, hide_index=True)
    for img, caption in [("shap_summary.png", "SHAP summary"),
                         ("model_confusion_matrix.png", "Confusion matrix")]:
        if (PLOTS_DIR / img).exists():
            st.image(str(PLOTS_DIR / img), caption=caption)

with tab_live:
    st.subheader("Live Prometheus counters from this service")
    st.caption("Raw metrics: http://localhost:8000/metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Successful requests", int(metric_value(
        "ml_prediction_requests_total", {"status": "success"})))
    k2.metric("Failed requests", int(metric_value(
        "ml_prediction_requests_total", {"status": "error"})))
    k3.metric("Fraud predictions", int(metric_value(
        "ml_predictions_total", {"prediction_class": "Fraud"})))
    k4.metric("Legitimate predictions", int(metric_value(
        "ml_predictions_total", {"prediction_class": "Legitimate"})))
    count = metric_value("ml_prediction_latency_seconds_count")
    total = metric_value("ml_prediction_latency_seconds_sum")
    st.metric("Average prediction latency", f"{(total / count * 1000) if count else 0:.1f} ms")
    st.button("Refresh")
