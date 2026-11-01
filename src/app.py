"""Streamlit inference service for the UPI fraud model, instrumented with Prometheus.

Run from the project root:   python -m streamlit run src/app.py
Prometheus metrics are exposed on  http://localhost:8000/metrics
"""
import sys
import time
from datetime import date, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from prometheus_client import REGISTRY  # noqa: E402

from src import metrics, ui_theme  # noqa: E402
from src.config import PLOTS_DIR, SERVING_DIR  # noqa: E402
from src.inference import (  # noqa: E402
    KYC_LABELS, REQUIRED_FIELDS, FraudPredictor, ValidationError,
)

st.set_page_config(page_title="UPI Fraud Shield | MLOps Exp 8", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")

SAMPLES = {
    "Legitimate sample": dict(
        Account_Age_Days=1200, User_Age=34, User_City="Mumbai", KYC="Verified",
        Transaction_Amount=850, Transaction_Hour=14, Transaction_Type="P2P",
        Merchant_Category="Grocery", Device_Type="Android", Device_Change=False,
        App_Version="v5.3.0", Previous_Transaction_Count=420, Avg_Transaction_Value=900.0,
        Transactions_Last_24h=2, Login_Attempts=1, IP_Match=True, Suspicious_Device=False,
        Rapid_Txn=False, Wallet_Balance=18000.0, Bank_Balance=240000.0),
    "Fraud-like sample": dict(
        Account_Age_Days=9, User_Age=22, User_City="Delhi", KYC="Pending",
        Transaction_Amount=48000, Transaction_Hour=2, Transaction_Type="Merchant",
        Merchant_Category="E-Commerce", Device_Type="Web Browser", Device_Change=True,
        App_Version="v4.8.5", Previous_Transaction_Count=3, Avg_Transaction_Value=1200.0,
        Transactions_Last_24h=9, Login_Attempts=5, IP_Match=False, Suspicious_Device=True,
        Rapid_Txn=True, Wallet_Balance=2500.0, Bank_Balance=52000.0),
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
        "Device_Type": form["Device_Type"], "Device_Change_Flag": int(form["Device_Change"]),
        "App_Version": form["App_Version"],
        "Previous_Transaction_Count": form["Previous_Transaction_Count"],
        "Avg_Transaction_Value": form["Avg_Transaction_Value"],
        "Transactions_Last_24h": form["Transactions_Last_24h"],
        "Login_Attempts": form["Login_Attempts"], "IP_Location_Match": int(form["IP_Match"]),
        "Suspicious_Device": int(form["Suspicious_Device"]),
        "Rapid_Transaction_Flag": int(form["Rapid_Txn"]),
        "Wallet_Balance": form["Wallet_Balance"], "Bank_Balance": form["Bank_Balance"],
    }


def load_sample(name):
    for key, value in SAMPLES[name].items():
        st.session_state[f"f_{key}"] = value
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_error", None)


def metric_value(name, labels=None):
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def clamp(x):
    return max(0.0, min(1.0, x))


def risk_signals(f):
    """Quick, rule-of-thumb signals shown beside the form (the model gives the real score)."""
    odd_hour = 1.0 if f["Transaction_Hour"] in (0, 1, 2, 3, 4, 22, 23) else 0.08
    anomalies = (int(f["Device_Change"]) + int(not f["IP_Match"]) + int(f["Suspicious_Device"]) + int(f["Rapid_Txn"])) / 4
    kyc = {"Verified": 0.05, "Pending": 0.6, "Not Submitted": 0.95}[f["KYC"]]
    return [
        ("New account", clamp(1 - f["Account_Age_Days"] / 365)),
        ("Amount vs usual spend", clamp(f["Transaction_Amount"] / (f["Avg_Transaction_Value"] + 1) / 6)),
        ("Unusual hour", odd_hour),
        ("Device / IP anomalies", anomalies),
        ("Velocity (last 24h)", clamp(f["Transactions_Last_24h"] / 10)),
        ("Login attempts", clamp((f["Login_Attempts"] - 1) / 4)),
        ("KYC risk", kyc),
    ]


# ------------------------------------------------------------------ setup
predictor = get_predictor()
opts = predictor.options["categories"]
rng = predictor.options["ranges"]
for k, v in SAMPLES["Legitimate sample"].items():
    st.session_state.setdefault(f"f_{k}", v)
st.session_state.setdefault("f_Transaction_Date", date.today())
st.session_state.setdefault("history", [])
ss = st.session_state

live = {
    "total": int(metric_value("ml_prediction_requests_total", {"status": "success"})),
    "fraud": int(metric_value("ml_predictions_total", {"prediction_class": "Fraud"})),
}
st.markdown(ui_theme.CSS, unsafe_allow_html=True)
st.markdown(ui_theme.hero_html(predictor.meta, live), unsafe_allow_html=True)

tab_predict, tab_perf, tab_live = st.tabs(["Predict", "Model insights", "Live monitoring"])

# ================================================================ PREDICT
with tab_predict:
    st.markdown(ui_theme.section(
        "Check a transaction", "Decide in milliseconds", "whether a payment should go through",
        "Move the controls, watch the risk signals react, then press Predict to get the CatBoost model's fraud probability."),
        unsafe_allow_html=True)
    b1, b2, _ = st.columns([1.1, 1.2, 4])
    b1.button("Load legitimate sample", on_click=load_sample, args=("Legitimate sample",))
    b2.button("Load fraud-like sample", on_click=load_sample, args=("Fraud-like sample",))

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        with st.container(border=True, key="card_inputs"):
            st.markdown('<div class="card-title">Transaction details <em>inputs</em></div>', unsafe_allow_html=True)
            a, b, c = st.columns(3)
            with a:
                st.markdown("**Transaction**")
                st.slider("Transaction amount (₹)", 0, 100000, step=50, key="f_Transaction_Amount")
                st.slider("Transaction hour (0-23)", 0, 23, key="f_Transaction_Hour")
                st.selectbox("Transaction type", opts["Transaction_Type"], key="f_Transaction_Type")
                st.selectbox("Merchant category (merchant payments)", opts["Merchant_Category"], key="f_Merchant_Category")
                st.date_input("Transaction date", key="f_Transaction_Date")
                st.slider("Transactions in last 24h", 0, 30, key="f_Transactions_Last_24h")
            with b:
                st.markdown("**User & account**")
                st.slider("User age", 18, 64, key="f_User_Age")
                st.selectbox("City", opts["User_City"], key="f_User_City")
                st.slider("Account age (days)", 0, 1825, step=1, key="f_Account_Age_Days")
                st.selectbox("KYC status", list(KYC_LABELS), key="f_KYC")
                st.number_input("Previous transactions", 0, 100000, step=10, key="f_Previous_Transaction_Count")
                st.number_input("Average transaction value (₹)", 0.0, 1_000_000.0, step=100.0, key="f_Avg_Transaction_Value")
                st.number_input("Wallet balance (₹)", 0.0, 10_000_000.0, step=500.0, key="f_Wallet_Balance")
                st.number_input("Bank balance (₹)", 0.0, 100_000_000.0, step=1000.0, key="f_Bank_Balance")
            with c:
                st.markdown("**Device & security**")
                st.selectbox("Device type", opts["Device_Type"], key="f_Device_Type")
                st.selectbox("App version", opts["App_Version"], key="f_App_Version")
                st.slider("Login attempts", 0, 10, key="f_Login_Attempts")
                st.toggle("Device changed", key="f_Device_Change")
                st.toggle("IP location matches", key="f_IP_Match")
                st.toggle("Suspicious device", key="f_Suspicious_Device")
                st.toggle("Rapid transaction flag", key="f_Rapid_Txn")
            predict_clicked = st.button("Predict fraud risk", type="primary")

    form = {k[2:]: v for k, v in ss.items() if k.startswith("f_")}
    if predict_clicked:
        result, error = score(predictor, to_inputs(form))
        ss["last_result"], ss["last_error"] = result, error
        if result:
            ss["history"].insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "amount": form["Transaction_Amount"],
                                     "label": result["prediction"], "p": result["fraud_probability"]})
            ss["history"] = ss["history"][:8]

    with right:
        with st.container(border=True, key="risk_card"):
            st.markdown(ui_theme.signals_html(risk_signals(form)), unsafe_allow_html=True)
            if ss.get("last_error"):
                st.error(f"Request rejected: {ss['last_error']}")
            elif ss.get("last_result"):
                st.markdown(ui_theme.verdict_html(ss["last_result"]), unsafe_allow_html=True)
            else:
                st.markdown('<div class="note" style="margin-top:14px">No prediction yet - press <b>Predict fraud risk</b>.</div>',
                            unsafe_allow_html=True)
            st.markdown(ui_theme.recent_html(ss["history"]), unsafe_allow_html=True)

    st.markdown(ui_theme.section("How it works", "From a raw payment", "to a monitored decision"), unsafe_allow_html=True)
    st.markdown(ui_theme.steps_html(), unsafe_allow_html=True)

# ================================================================ INSIGHTS
with tab_perf:
    st.markdown(ui_theme.section(
        "Model insights", "Why you can trust the score", "14 model runs, one winner",
        "Seven classifiers were trained with Class Weights and with SMOTE; the best by F1-Macro then ROC-AUC is deployed."),
        unsafe_allow_html=True)
    mm = predictor.meta["metrics"]
    st.markdown(ui_theme.tiles_html([
        ("Accuracy", f"{mm['Accuracy']:.3f}", ""), ("Precision", f"{mm['Precision']:.3f}", ""),
        ("Recall", f"{mm['Recall']:.3f}", "g"), ("F1-Fraud", f"{mm['F1-Fraud']:.3f}", ""),
        ("F1-Macro", f"{mm['F1-Macro']:.3f}", "g"), ("ROC-AUC", f"{mm['ROC-AUC']:.3f}", "g"),
        ("PR-AUC", f"{mm['PR-AUC']:.3f}", ""),
    ]), unsafe_allow_html=True)
    lb = pd.read_csv(SERVING_DIR / "model_leaderboard.csv")
    col1, col2 = st.columns([1, 1.15], gap="large")
    with col1:
        with st.container(border=True, key="card_bars"):
            st.markdown('<div class="card-title">F1-Macro by model <em>leaderboard</em></div>', unsafe_allow_html=True)
            items = [(f"{r['Model']} · {r['Method']}", float(r["F1-Macro"])) for _, r in lb.head(8).iterrows()]
            st.markdown(ui_theme.hbars_html(items, best=items[0][0]), unsafe_allow_html=True)
    with col2:
        with st.container(border=True, key="card_board"):
            st.markdown('<div class="card-title">Full leaderboard <em>all 14 runs</em></div>', unsafe_allow_html=True)
            st.dataframe(lb.round(4), width="stretch", height=330)
    i1, i2 = st.columns(2, gap="large")
    for col, img, title in [(i1, "model_confusion_matrix.png", "Confusion matrix"), (i2, "shap_summary.png", "SHAP summary")]:
        with col, st.container(border=True, key=f"card_{img[:5]}"):
            st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
            if (PLOTS_DIR / img).exists():
                st.image(str(PLOTS_DIR / img), width="stretch")

# ================================================================ LIVE
with tab_live:
    st.markdown(ui_theme.section(
        "Live monitoring", "What Prometheus sees", "right now",
        "These counters are the same series scraped from http://localhost:8000/metrics and graphed in Grafana."),
        unsafe_allow_html=True)
    ok = int(metric_value("ml_prediction_requests_total", {"status": "success"}))
    bad = int(metric_value("ml_prediction_requests_total", {"status": "error"}))
    fr = int(metric_value("ml_predictions_total", {"prediction_class": "Fraud"}))
    lg = int(metric_value("ml_predictions_total", {"prediction_class": "Legitimate"}))
    count = metric_value("ml_prediction_latency_seconds_count")
    total = metric_value("ml_prediction_latency_seconds_sum")
    st.markdown(ui_theme.tiles_html([
        ("Successful requests", f"{ok:,}", "g"), ("Failed requests", f"{bad:,}", "r" if bad else ""),
        ("Fraud predictions", f"{fr:,}", "r" if fr else ""), ("Legitimate predictions", f"{lg:,}", "g"),
        ("Avg prediction latency", f"{(total / count * 1000) if count else 0:.1f} ms", ""),
    ]), unsafe_allow_html=True)
    l1, l2 = st.columns([1.2, 1], gap="large")
    with l1:
        with st.container(border=True, key="card_sim"):
            st.markdown('<div class="card-title">Traffic simulator <em>replays real transactions</em></div>', unsafe_allow_html=True)
            n_sim = st.slider("Transactions to simulate", 5, 100, 25, step=5)
            if st.button("▶ Simulate traffic", type="primary"):
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
                           f"{counts['Legitimate']} legitimate, {counts['errors']} errors - press Refresh to update the tiles.")
            st.button("Refresh counters")
    with l2:
        with st.container(border=True, key="card_class"):
            st.markdown('<div class="card-title">Predictions by class <em>live</em></div>', unsafe_allow_html=True)
            class_bars = [("Legitimate", float(lg)), ("Fraud", float(fr)), ("Success", float(ok)), ("Errors", float(bad))]
            st.markdown(ui_theme.hbars_html(class_bars, integer=True), unsafe_allow_html=True)

st.markdown(ui_theme.trust_html(), unsafe_allow_html=True)
st.markdown(ui_theme.footer_html(), unsafe_allow_html=True)
