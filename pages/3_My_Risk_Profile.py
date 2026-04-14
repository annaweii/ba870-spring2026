import streamlit as st
import pandas as pd
import numpy as np

from utils import (
    apply_global_styles,
    render_header,
    load_artifacts,
    get_risk_label,
    fmt
)

st.set_page_config(page_title="Custom Risk Comparison", layout="wide")
apply_global_styles()

model, features, THRESHOLD, _ = load_artifacts()

render_header(
    "Custom Risk Comparison",
    "Input company financials to estimate distress risk",
    "header-method"
)

# -------------------------------
# INPUT FORM
# -------------------------------
st.markdown("#### Enter Company Financials")

col1, col2, col3 = st.columns(3)

with col1:
    act = st.number_input("Current Assets", value=1000.0)
    at = st.number_input("Total Assets", value=2000.0)
    lt = st.number_input("Total Liabilities", value=1200.0)
    wcap = st.number_input("Working Capital", value=200.0)

with col2:
    ni = st.number_input("Net Income", value=100.0)
    ebit = st.number_input("EBIT", value=150.0)
    sale = st.number_input("Revenue", value=2500.0)
    re = st.number_input("Retained Earnings", value=300.0)

with col3:
    oancf = st.number_input("Operating Cash Flow", value=120.0)
    mkvalt = st.number_input("Market Value", value=5000.0)
    current_ratio = st.number_input("Current Ratio", value=1.5)
    debt_to_assets = st.number_input("Debt to Assets", value=0.5)

run = st.button("Analyze Custom Company", use_container_width=True)

# -------------------------------
# RUN MODEL
# -------------------------------
if run:

    # Derived features (same logic as utils)
    liabilities_to_assets = lt / at if at != 0 else np.nan
    roa = ni / at if at != 0 else np.nan
    ocf_to_liabilities = oancf / lt if lt != 0 else np.nan

    A_wc_to_assets = wcap / at if at != 0 else np.nan
    B_re_to_assets = re / at if at != 0 else np.nan
    C_ebit_to_assets = ebit / at if at != 0 else np.nan
    D_mve_to_lt = mkvalt / lt if lt != 0 else np.nan
    E_sales_to_assets = sale / at if at != 0 else np.nan

    Altman_Z = (
        1.2 * A_wc_to_assets +
        1.4 * B_re_to_assets +
        3.3 * C_ebit_to_assets +
        0.6 * D_mve_to_lt +
        1.0 * E_sales_to_assets
    )

    # Build feature dict
    feat_dict = {
        "Altman_Z": Altman_Z,
        "A_wc_to_assets": A_wc_to_assets,
        "B_re_to_assets": B_re_to_assets,
        "C_ebit_to_assets": C_ebit_to_assets,
        "D_mve_to_lt": D_mve_to_lt,
        "E_sales_to_assets": E_sales_to_assets,
        "act": act,
        "at": at,
        "lt": lt,
        "wcap": wcap,
        "ebit": ebit,
        "ni": ni,
        "sale": sale,
        "revt": sale,
        "oancf": oancf,
        "mkvalt": mkvalt,
        "liabilities_to_assets": liabilities_to_assets,
        "roa": roa,
        "ocf_to_liabilities": ocf_to_liabilities,
        "current_ratio": current_ratio,
        "debt_to_assets": debt_to_assets,
    }

    # Align with model features
    X = pd.DataFrame([feat_dict]).reindex(columns=features, fill_value=np.nan)

    prob = float(model.predict_proba(X)[0][1])
    risk_label, badge_class, banner_class = get_risk_label(prob, THRESHOLD)

    # -------------------------------
    # OUTPUT
    # -------------------------------
    st.markdown(f"""
    <div class="{banner_class}">
        <div class="banner-title">{risk_label}</div>
        <div class="banner-sub">Distress Probability: {prob*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3 = st.columns(3)
    k1.metric("Distress Probability", f"{prob*100:.1f}%")
    k2.metric("Altman Z-Score", f"{Altman_Z:.2f}")
    k3.metric("Threshold", f"{THRESHOLD:.2f}")
