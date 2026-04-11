import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

# -- Load model artifacts ------------------------------------------------------
model      = joblib.load("final_app_model.joblib")
features   = json.load(open("final_app_features.json"))
thresh_cfg = json.load(open("final_app_threshold.json"))
THRESHOLD  = thresh_cfg["threshold"]
metrics    = pd.read_csv("final_model_metrics.csv").set_index("metric")["value"]

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="Corporate Bankruptcy Risk Analyzer",
    layout="wide"
)

# -- Navigation ----------------------------------------------------------------
pages = [
    "Home",
    "Company Risk Checker",
    "Financial Health Dashboard",
    "Peer Comparison",
    "Model Insights"
]
page = st.sidebar.radio("Navigate", pages)

# ==============================================================================
# PAGE 1 - HOME
# ==============================================================================
if page == "Home":
    st.title("Corporate Bankruptcy Risk Analyzer")
    st.subheader("An early-warning tool for financial distress prediction")

    st.markdown("""
    This app uses a machine learning model trained on real financial data to estimate
    the probability that a publicly traded company will become financially distressed
    within the **next 12 months**.
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "HistGradientBoosting")
    col2.metric("ROC-AUC", f"{float(metrics['roc_auc']):.4f}")
    col3.metric("Decision Threshold", f"{float(metrics['best_threshold']):.4f}")

    st.divider()

    st.markdown("### How to use this app")
    st.markdown("""
    - **Company Risk Checker** — Enter any stock ticker to get an instant distress probability score
    - **Financial Health Dashboard** — Explore the company's key financial ratios and trends
    - **Peer Comparison** — Compare a company's risk profile against similar firms
    - **Model Insights** — Understand how the model works and what drives its predictions
    """)

    st.info("Start by entering a ticker in the Company Risk Checker tab on the left.")

# ==============================================================================
# PAGE 2 - COMPANY RISK CHECKER (placeholder)
# ==============================================================================
elif page == "Company Risk Checker":
    st.title("Company Risk Checker")
    st.info("Coming soon — we will build this next.")

# ==============================================================================
# PAGE 3 - FINANCIAL HEALTH DASHBOARD (placeholder)
# ==============================================================================
elif page == "Financial Health Dashboard":
    st.title("Financial Health Dashboard")
    st.info("Coming soon.")

# ==============================================================================
# PAGE 4 - PEER COMPARISON (placeholder)
# ==============================================================================
elif page == "Peer Comparison":
    st.title("Peer Comparison")
    st.info("Coming soon.")

# ==============================================================================
# PAGE 5 - MODEL INSIGHTS (placeholder)
# ==============================================================================
elif page == "Model Insights":
    st.title("Model Insights and Methodology")
    st.info("Coming soon.")
