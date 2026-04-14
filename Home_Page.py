import streamlit as st
from utils import apply_global_styles, render_header, kpi, load_artifacts, render_sidebar_header

# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(page_title="RiskMonitor", layout="wide")

# Apply global styles
apply_global_styles()
render_sidebar_header()

# Load model metadata
_, _, _, metrics = load_artifacts()

# ------------------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------------------
render_header(
    "RiskMonitor",
    "Early-warning financial distress prediction for public companies",
    "header-home"
)

# ------------------------------------------------------------------------------
# ABOUT SECTION
# ------------------------------------------------------------------------------
st.markdown("#### What This App Does")
st.markdown("""
<div class="card">
RiskMonitor estimates the probability that a publicly traded company will become
financially distressed within the next 12 months. It combines a machine learning model
trained on historical accounting data with live financial and market signals pulled
directly from public sources. The result is an interpretable risk score grounded in
both classical finance theory and modern predictive modeling.
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# WHY IT MATTERS
# ------------------------------------------------------------------------------
st.markdown("#### Why It Matters")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
    <b>Investors</b> — Screen portfolio holdings for early warning signals before
    distress becomes visible in price action.<br><br>

    <b>Lenders</b> — Identify deteriorating credit quality using forward-looking
    indicators beyond traditional credit metrics.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <b>Analysts</b> — Compare companies within a sector on a consistent
    risk framework combining accounting and market data.<br><br>

    <b>Students</b> — Explore how machine learning applies to classical
    financial distress modeling in a hands-on environment.
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# HOW TO USE
# ------------------------------------------------------------------------------
st.markdown("#### How To Use")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="step-box">
        <div class="step-num">01</div>
        <div class="step-text">
            Open the Company Risk Check page and enter a public stock ticker
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="step-box">
        <div class="step-num">02</div>
        <div class="step-text">
            Review the distress probability, risk category, and explanation
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="step-box">
        <div class="step-num">03</div>
        <div class="step-text">
            Use the dashboard and insights pages for deeper analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# MODEL HIGHLIGHTS
# ------------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### Model Highlights")

m1, m2, m3, m4 = st.columns([1.2, 1, 1, 1])

m1.markdown(
    kpi("Model", "HistGradientBoosting", "kpi-box-blue"),
    unsafe_allow_html=True
)

m2.markdown(
    kpi("ROC-AUC", f"{float(metrics['roc_auc']):.4f}", "kpi-box-teal"),
    unsafe_allow_html=True
)

m3.markdown(
    kpi("Target", "Next-Year Distress", "kpi-box-purple"),
    unsafe_allow_html=True
)

m4.markdown(
    kpi("Benchmark", "Altman Z-Score", "kpi-box-slate"),
    unsafe_allow_html=True
)

# ------------------------------------------------------------------------------
# DISCLAIMER
# ------------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer">
This tool is for academic and analytical purposes only. It is not investment,
lending, or legal advice. Predictions are probabilistic and based on publicly
available data that may be incomplete or delayed.
</div>
""", unsafe_allow_html=True)
