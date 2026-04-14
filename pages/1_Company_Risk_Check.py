import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import (
    apply_global_styles,
    render_header,
    kpi,
    fmt,
    load_artifacts,
    fetch_company_data,
    get_risk_label,
    get_altman_label,
)

st.set_page_config(page_title="Company Risk Check", layout="wide")
apply_global_styles()

model, features, THRESHOLD, _ = load_artifacts()

render_header(
    "Company Risk Check",
    "Enter a ticker to generate a full distress risk profile",
    "header-risk"
)

col_input, _ = st.columns([1, 3])
with col_input:
    ticker = st.text_input("Stock Ticker", placeholder="e.g. AAPL, F, GME").strip().upper()
    run = st.button("Analyze", use_container_width=True)

if run and ticker:
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            feat_dict, company_name, sector, industry, altman_z, info, hist = fetch_company_data(ticker)
        except Exception as e:
            st.error(f"Could not retrieve data for {ticker}: {e}")
            st.stop()

    X = pd.DataFrame([feat_dict]).reindex(columns=features, fill_value=np.nan)
    prob = float(model.predict_proba(X)[0][1])
    risk_label, badge_class, banner_class = get_risk_label(prob, THRESHOLD)
    z_label, z_badge = get_altman_label(altman_z)

    st.markdown(f"""
    <div class="company-strip">
        <div style="font-size:1.55rem; font-weight:800; color:#0f172a;">{company_name}</div>
        <div style="color:#64748b; font-size:0.95rem; margin-top:0.2rem;">{sector} &nbsp;—&nbsp; {industry}</div>
    </div>
    """, unsafe_allow_html=True)

    banner_sub = (
        f"Model probability: {prob*100:.1f}% — above the distress threshold of {THRESHOLD*100:.1f}%"
        if prob >= THRESHOLD * 0.8
        else f"Model probability: {prob*100:.1f}% — below the distress threshold of {THRESHOLD*100:.1f}%"
    )
    st.markdown(f"""
    <div class="{banner_class}">
        <div class="banner-title">{risk_label}</div>
        <div class="banner-sub">{banner_sub}</div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Distress Probability", f"{prob*100:.1f}%"), unsafe_allow_html=True)
    k2.markdown(
        f'<div class="kpi-box"><div class="kpi-label">Risk Category</div><div class="kpi-value" style="font-size:1.05rem; margin-top:0.5rem;"><span class="{badge_class}">{risk_label}</span></div></div>',
        unsafe_allow_html=True
    )
    k3.markdown(kpi("Model Threshold", f"{THRESHOLD:.4f}"), unsafe_allow_html=True)
    k4.markdown(
        f'<div class="kpi-box"><div class="kpi-label">Altman Z-Score</div><div class="kpi-value">{fmt(altman_z)}<br><span class="{z_badge}" style="font-size:0.72rem;">{z_label}</span></div></div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    g_col, s_col = st.columns([1.1, 1])
    with g_col:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 38, "color": "#0f172a"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#dc2626" if prob >= THRESHOLD else "#16a34a", "thickness": 0.28},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, THRESHOLD * 80], "color": "#f0fdf4"},
                    {"range": [THRESHOLD * 80, THRESHOLD * 130], "color": "#fff7ed"},
                    {"range": [THRESHOLD * 130, 100], "color": "#fef2f2"},
                ],
                "threshold": {
                    "line": {"color": "#1e293b", "width": 3},
                    "thickness": 0.75,
                    "value": THRESHOLD * 100
                }
            },
            title={"text": "Distress Probability", "font": {"size": 15, "color": "#64748b"}}
        ))
        gauge.update_layout(
            height=300,
            margin=dict(t=50, b=10, l=30, r=30),
            paper_bgcolor="white",
            font={"family": "Inter, Segoe UI, sans-serif"}
        )
        st.plotly_chart(gauge, use_container_width=True)

    with s_col:
        st.markdown("<br>", unsafe_allow_html=True)
        drivers = []
        if pd.notna(feat_dict.get("roa")) and feat_dict["roa"] < 0:
            drivers.append("negative profitability (ROA below zero)")
        if pd.notna(feat_dict.get("liabilities_to_assets")) and feat_dict["liabilities_to_assets"] > 0.7:
            drivers.append("high leverage (liabilities over 70% of assets)")
        if pd.notna(feat_dict.get("current_ratio")) and feat_dict["current_ratio"] < 1:
            drivers.append("weak liquidity (current ratio below 1)")
        if pd.notna(feat_dict.get("ocf_to_liabilities")) and feat_dict["ocf_to_liabilities"] < 0:
            drivers.append("negative operating cash flow relative to liabilities")
        if pd.notna(altman_z) and altman_z < 1.81:
            drivers.append("an Altman Z-score in the distress zone")

        if prob >= THRESHOLD:
            summary = "This company shows elevated next-year distress risk" + (f", driven by {', '.join(drivers)}." if drivers else ".")
        else:
            summary = "Based on current financial and market data, this company does not show elevated distress risk relative to the model threshold."

        st.markdown("**Plain-English Summary**")
        st.markdown(f'<div class="highlight-card">{summary}</div>', unsafe_allow_html=True)

        st.markdown("**Benchmark Signals**")
        bb1, bb2 = st.columns(2)
        bb1.markdown(kpi("Altman Z-Score", fmt(altman_z)), unsafe_allow_html=True)
        bb2.markdown(kpi("Current Ratio", fmt(feat_dict.get("current_ratio"))), unsafe_allow_html=True)
        bb3, bb4 = st.columns(2)
        bb3.markdown(kpi("ROA", fmt(feat_dict.get("roa"), pct=True)), unsafe_allow_html=True)
        bb4.markdown(kpi("Debt to Assets", fmt(feat_dict.get("debt_to_assets"), pct=True)), unsafe_allow_html=True)

    st.session_state.update({
        "last_ticker": ticker,
        "last_feat_dict": feat_dict,
        "last_prob": prob,
        "last_altman": altman_z,
        "last_company": company_name,
        "last_hist": hist,
    })
