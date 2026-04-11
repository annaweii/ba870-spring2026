import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------------------------
# LOAD MODEL ARTIFACTS
# ------------------------------------------------------------------------------
model      = joblib.load("final_app_model.joblib")
features   = json.load(open("final_app_features.json"))
thresh_cfg = json.load(open("final_app_threshold.json"))
THRESHOLD  = thresh_cfg["threshold"]
metrics    = pd.read_csv("final_model_metrics.csv").set_index("metric")["value"]

# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(page_title="RiskMonitor", layout="wide")

# ------------------------------------------------------------------------------
# STYLING
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background-color: #f4f7fb;
    }

    /* Sidebar */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #061225 0%, #0b1f45 100%);
        border-right: 1px solid #18345f;
    }
    div[data-testid="stSidebar"] * {
        color: #d9e3f0 !important;
    }
    div[data-testid="stSidebar"] h2 {
        color: #ffffff !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        letter-spacing: 1px;
    }

    /* Main container */
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }

    /* Shared header shell */
    .page-header {
        padding: 2.5rem 2.8rem 2rem 2.8rem;
        border-radius: 18px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.16);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .page-header::before {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        right: -60px;
        top: -70px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.14) 0%, transparent 70%);
    }
    .page-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.4px;
    }
    .page-header p {
        color: rgba(255,255,255,0.82);
        font-size: 1rem;
        margin-top: 0.45rem;
    }

    .header-home {
        background: linear-gradient(135deg, #081223 0%, #0f2454 55%, #1d4ed8 100%);
    }
    .header-risk {
        background: linear-gradient(135deg, #0b132b 0%, #1e3a8a 55%, #2563eb 100%);
    }
    .header-dashboard {
        background: linear-gradient(135deg, #082f49 0%, #0f766e 55%, #14b8a6 100%);
    }
    .header-method {
        background: linear-gradient(135deg, #1e293b 0%, #334155 55%, #64748b 100%);
    }

    /* Cards */
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.45rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        line-height: 1.72;
        color: #334155;
    }

    .highlight-card {
        background: linear-gradient(135deg, #eff6ff, #ffffff);
        border: 1px solid #bfdbfe;
        border-left: 4px solid #3b82f6;
        border-radius: 15px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        color: #334155;
        line-height: 1.7;
    }

    .company-strip {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #dbe7f3;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }

    /* Risk banners */
    .banner-high {
        background: linear-gradient(135deg, #7f1d1d, #dc2626);
        color: #ffffff;
        padding: 1.3rem 1.6rem;
        border-radius: 14px;
        margin-bottom: 1.3rem;
        box-shadow: 0 8px 20px rgba(220,38,38,0.22);
    }
    .banner-moderate {
        background: linear-gradient(135deg, #9a3412, #ea580c);
        color: #ffffff;
        padding: 1.3rem 1.6rem;
        border-radius: 14px;
        margin-bottom: 1.3rem;
        box-shadow: 0 8px 20px rgba(234,88,12,0.20);
    }
    .banner-low {
        background: linear-gradient(135deg, #14532d, #16a34a);
        color: #ffffff;
        padding: 1.3rem 1.6rem;
        border-radius: 14px;
        margin-bottom: 1.3rem;
        box-shadow: 0 8px 20px rgba(22,163,74,0.20);
    }
    .banner-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: 0.2px;
    }
    .banner-sub {
        font-size: 0.95rem;
        opacity: 0.9;
        margin-top: 0.25rem;
    }

    /* KPI boxes */
    .kpi-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.15rem 1rem;
        text-align: center;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    .kpi-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    }
    .kpi-box .kpi-label {
        color: #94a3b8;
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.3px;
    }
    .kpi-box .kpi-value {
        color: #0f172a;
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 0.35rem;
        letter-spacing: -0.4px;
    }

    /* Badges */
    .badge-high {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        color: #b91c1c;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.84rem;
        display: inline-block;
    }
    .badge-moderate {
        background: #fff7ed;
        border: 1px solid #fdba74;
        color: #c2410c;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.84rem;
        display: inline-block;
    }
    .badge-low {
        background: #f0fdf4;
        border: 1px solid #86efac;
        color: #166534;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.84rem;
        display: inline-block;
    }

    /* Step boxes */
    .step-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #3b82f6;
        border-radius: 16px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }
    .step-box .step-num {
        color: #2563eb;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .step-box .step-text {
        color: #475569;
        font-size: 0.9rem;
        margin-top: 0.45rem;
        line-height: 1.55;
    }

    /* Disclaimer */
    .disclaimer {
        background: linear-gradient(135deg, #eff6ff, #f8fbff);
        border-left: 4px solid #60a5fa;
        border-radius: 12px;
        padding: 1rem 1.3rem;
        color: #475569;
        font-size: 0.84rem;
        margin-top: 2rem;
        line-height: 1.65;
    }

    /* Section styles */
    .section-divider {
        border: none;
        border-top: 2px solid #e2e8f0;
        margin: 2rem 0 1.3rem 0;
    }
    .section-title {
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        color: #94a3b8;
        margin-bottom: 0.8rem;
    }

    h3, h4 {
        color: #0f172a;
        font-weight: 700;
    }
    h4 {
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Analyze button */
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb, #3b82f6);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        padding: 0.55rem 1.3rem;
        box-shadow: 0 6px 16px rgba(37,99,235,0.28);
        transition: opacity 0.2s ease, transform 0.15s ease;
    }
    .stButton > button:hover {
        opacity: 0.94;
        transform: translateY(-1px);
        color: #ffffff;
    }

    .stTextInput input {
        border-radius: 10px;
        border: 1.5px solid #cbd5e1;
        font-size: 1rem;
        padding: 0.55rem 1rem;
        background: #ffffff;
    }
    .stTextInput input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.14);
    }

    /* Driver cards */
    .driver-grid-title {
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        color: #94a3b8;
        margin: 0.35rem 0 0.9rem 0;
    }

    .driver-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #dbe7f3;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }

    .driver-card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.8rem;
    }

    .driver-card-name {
        color: #0f172a;
        font-size: 0.98rem;
        font-weight: 700;
        line-height: 1.3;
    }

    .driver-card-values {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
    }

    .driver-mini-label {
        color: #94a3b8;
        font-size: 0.68rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.15rem;
    }

    .driver-mini-value {
        color: #1e293b;
        font-size: 1rem;
        font-weight: 700;
    }

    .signal-good {
        background: #ecfdf5;
        color: #166534;
        border: 1px solid #86efac;
        border-radius: 999px;
        padding: 0.28rem 0.72rem;
        font-size: 0.72rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .signal-bad {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        border-radius: 999px;
        padding: 0.28rem 0.72rem;
        font-size: 0.72rem;
        font-weight: 700;
        white-space: nowrap;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------------------
st.sidebar.markdown("## RiskMonitor")
st.sidebar.markdown("---")
page = st.sidebar.radio("", [
    "Home",
    "Company Risk Check",
    "Financial Health Dashboard",
    "Methodology"
])

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def safe_div(a, b):
    try:
        return a / b if b and b != 0 else np.nan
    except:
        return np.nan

def render_header(title, subtitle, header_class):
    st.markdown(
        f"""
        <div class="page-header {header_class}">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_data(show_spinner=False)
def fetch_company_data(ticker: str):
    t    = yf.Ticker(ticker)
    info = t.info or {}
    bs   = t.balance_sheet
    inc  = t.income_stmt
    cf   = t.cashflow
    hist = t.history(period="1y")

    def get(sheet, *rows):
        for row in rows:
            try:
                val = sheet.loc[row].iloc[0]
                if pd.notna(val):
                    return float(val)
            except:
                pass
        return np.nan

    act   = get(bs,  "Current Assets")
    at    = get(bs,  "Total Assets")
    lt    = get(bs,  "Total Liabilities Net Minority Interest", "Total Liabilities")
    lct   = get(bs,  "Current Liabilities")
    wcap  = act - lct if pd.notna(act) and pd.notna(lct) else np.nan
    ebit  = get(inc, "EBIT", "Operating Income")
    ni    = get(inc, "Net Income")
    sale  = get(inc, "Total Revenue")
    oancf = get(cf,  "Operating Cash Flow", "Cash From Operations")
    xint  = get(inc, "Interest Expense")
    re    = get(bs,  "Retained Earnings")
    dltt  = get(bs,  "Long Term Debt")
    dlc   = get(bs,  "Current Debt", "Short Term Debt")

    mkvalt     = info.get("marketCap", np.nan)
    prcc_f     = info.get("previousClose", np.nan)
    last_close = prcc_f

    if len(hist) > 0:
        closes = hist["Close"]
        ret_1m       = float(closes.iloc[-1] / closes.iloc[max(-21,  -len(closes))] - 1) if len(closes) >= 5 else np.nan
        ret_3m       = float(closes.iloc[-1] / closes.iloc[max(-63,  -len(closes))] - 1) if len(closes) >= 20 else np.nan
        ret_6m       = float(closes.iloc[-1] / closes.iloc[max(-126, -len(closes))] - 1) if len(closes) >= 40 else np.nan
        ret_12m      = float(closes.iloc[-1] / closes.iloc[0]) - 1
        daily_ret    = closes.pct_change().dropna()
        vol_30d      = float(daily_ret.iloc[-30:].std() * np.sqrt(252)) if len(daily_ret) >= 20 else np.nan
        vol_90d      = float(daily_ret.iloc[-90:].std() * np.sqrt(252)) if len(daily_ret) >= 60 else np.nan
        vol_252d     = float(daily_ret.std() * np.sqrt(252))
        peak         = closes.cummax()
        drawdown_1y  = float(((closes - peak) / peak).min())
        avg_volume_30d = float(hist["Volume"].iloc[-30:].mean()) if "Volume" in hist.columns else np.nan
    else:
        ret_1m = ret_3m = ret_6m = ret_12m = np.nan
        vol_30d = vol_90d = vol_252d = np.nan
        drawdown_1y = avg_volume_30d = np.nan

    liabilities_to_assets = safe_div(lt, at)
    roa                   = safe_div(ni, at)
    ocf_to_liabilities    = safe_div(oancf, lt)
    current_ratio         = safe_div(act, lct)
    debt_to_assets        = safe_div((dltt or 0) + (dlc or 0), at)

    A_wc_to_assets    = safe_div(wcap, at)
    B_re_to_assets    = safe_div(re, at)
    C_ebit_to_assets  = safe_div(ebit, at)
    D_mve_to_lt       = safe_div(mkvalt, lt)
    E_sales_to_assets = safe_div(sale, at)

    Altman_Z = (
        1.2 * (A_wc_to_assets or 0) +
        1.4 * (B_re_to_assets or 0) +
        3.3 * (C_ebit_to_assets or 0) +
        0.6 * (D_mve_to_lt or 0) +
        1.0 * (E_sales_to_assets or 0)
    )

    feature_dict = {
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
        "xint": xint,
        "mkvalt": mkvalt,
        "prcc_f": prcc_f,
        "last_close": last_close,
        "ret_1m": ret_1m,
        "ret_3m": ret_3m,
        "ret_6m": ret_6m,
        "ret_12m": ret_12m,
        "vol_30d": vol_30d,
        "vol_90d": vol_90d,
        "vol_252d": vol_252d,
        "drawdown_1y": drawdown_1y,
        "avg_volume_30d": avg_volume_30d,
        "liabilities_to_assets": liabilities_to_assets,
        "roa": roa,
        "ocf_to_liabilities": ocf_to_liabilities,
        "current_ratio": current_ratio,
        "debt_to_assets": debt_to_assets,
    }

    return (
        feature_dict,
        info.get("longName", ticker.upper()),
        info.get("sector", "N/A"),
        info.get("industry", "N/A"),
        Altman_Z,
        info,
        hist
    )

def get_risk_label(prob):
    if prob >= THRESHOLD * 1.3:
        return "High Risk", "badge-high", "banner-high"
    elif prob >= THRESHOLD * 0.8:
        return "Moderate Risk", "badge-moderate", "banner-moderate"
    else:
        return "Low Risk", "badge-low", "banner-low"

def get_altman_label(z):
    if z < 1.81:
        return "Distress Zone", "badge-high"
    elif z < 2.99:
        return "Grey Zone", "badge-moderate"
    else:
        return "Safe Zone", "badge-low"

def fmt(val, pct=False, dollar=False, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if dollar:
        if abs(val) >= 1e9:
            return f"${val/1e9:.1f}B"
        if abs(val) >= 1e6:
            return f"${val/1e6:.1f}M"
        return f"${val:,.0f}"
    if pct:
        return f"{val*100:.1f}%"
    return f"{val:.{decimals}f}"

def kpi(label, value):
    return f'<div class="kpi-box"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'

# ==============================================================================
# PAGE 1 — HOME
# ==============================================================================
if page == "Home":
    render_header(
        "RiskMonitor",
        "Early-warning financial distress prediction for public companies",
        "header-home"
    )

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

    st.markdown("#### How To Use It")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="step-box">
            <div class="step-num">01</div>
            <div class="step-text">Enter any stock ticker on the Company Risk Check page</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="step-box">
            <div class="step-num">02</div>
            <div class="step-text">Review the distress probability, risk category, and plain-English explanation</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="step-box">
            <div class="step-num">03</div>
            <div class="step-text">Explore the Financial Health Dashboard for a full accounting and market breakdown</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Model Highlights")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(kpi("Model", "HistGradientBoosting"), unsafe_allow_html=True)
    m2.markdown(kpi("ROC-AUC", f"{float(metrics['roc_auc']):.4f}"), unsafe_allow_html=True)
    m3.markdown(kpi("Target", "Next-Year Distress"), unsafe_allow_html=True)
    m4.markdown(kpi("Benchmark", "Altman Z-Score"), unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
    This tool is for academic and analytical purposes only. It is not investment, lending,
    or legal advice. Predictions are probabilistic and based on publicly available data
    that may be incomplete or delayed.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 2 — COMPANY RISK CHECK
# ==============================================================================
elif page == "Company Risk Check":
    render_header(
        "Company Risk Check",
        "Enter a ticker to generate a full distress risk profile",
        "header-risk"
    )

    col_input, _ = st.columns([1, 3])
    with col_input:
        ticker = st.text_input("Stock Ticker", placeholder="e.g. AAPL, F, GME").strip().upper()
        run    = st.button("Analyze", use_container_width=True)

    if run and ticker:
        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                feat_dict, company_name, sector, industry, altman_z, info, hist = fetch_company_data(ticker)
            except Exception:
                st.error(f"Could not retrieve data for {ticker}. Check the ticker and try again.")
                st.stop()

        X    = pd.DataFrame([feat_dict])[features]
        prob = float(model.predict_proba(X)[0][1])
        risk_label, badge_class, banner_class = get_risk_label(prob)
        z_label, z_badge = get_altman_label(altman_z)

        # Company strip
        st.markdown(f"""
        <div class="company-strip">
            <div style="font-size:1.55rem; font-weight:800; color:#0f172a;">{company_name}</div>
            <div style="color:#64748b; font-size:0.95rem; margin-top:0.2rem;">{sector} &nbsp;—&nbsp; {industry}</div>
        </div>
        """, unsafe_allow_html=True)

        # Colored risk banner
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

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi("Distress Probability", f"{prob*100:.1f}%"), unsafe_allow_html=True)
        k2.markdown(
            f'<div class="kpi-box"><div class="kpi-label">Risk Category</div><div class="kpi-value" style="font-size:1.05rem; margin-top:0.5rem;"><span class="{badge_class}">{risk_label}</span></div></div>',
            unsafe_allow_html=True
        )
        k3.markdown(kpi("Model Threshold", f"{THRESHOLD:.4f}"), unsafe_allow_html=True)
        k4.markdown(
            f'<div class="kpi-box"><div class="kpi-label">Altman Z-Score</div><div class="kpi-value">{altman_z:.2f}<br><span class="{z_badge}" style="font-size:0.72rem;">{z_label}</span></div></div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge + summary
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
                        {"range": [0, THRESHOLD*80], "color": "#f0fdf4"},
                        {"range": [THRESHOLD*80, THRESHOLD*130], "color": "#fff7ed"},
                        {"range": [THRESHOLD*130, 100], "color": "#fef2f2"},
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
            if feat_dict.get("roa") and feat_dict["roa"] < 0:
                drivers.append("negative profitability (ROA below zero)")
            if feat_dict.get("liabilities_to_assets") and feat_dict["liabilities_to_assets"] > 0.7:
                drivers.append("high leverage (liabilities over 70% of assets)")
            if feat_dict.get("current_ratio") and feat_dict["current_ratio"] < 1:
                drivers.append("weak liquidity (current ratio below 1)")
            if feat_dict.get("ocf_to_liabilities") and feat_dict["ocf_to_liabilities"] < 0:
                drivers.append("negative operating cash flow relative to liabilities")
            if altman_z < 1.81:
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

        # Risk Drivers section
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("#### What Is Driving This Score")
        st.markdown("<p class='section-title'>Feature performance vs healthy benchmark</p>", unsafe_allow_html=True)

        benchmarks = {
            "roa":                   ("ROA", 0.05, True, False),
            "current_ratio":         ("Current Ratio", 1.5, False, False),
            "liabilities_to_assets": ("Liabilities to Assets", 0.5, False, True),
            "debt_to_assets":        ("Debt to Assets", 0.35, False, True),
            "ocf_to_liabilities":    ("OCF to Liabilities", 0.1, False, False),
            "A_wc_to_assets":        ("Working Capital / Assets", 0.1, False, False),
            "C_ebit_to_assets":      ("EBIT / Assets", 0.08, False, False),
            "Altman_Z":              ("Altman Z-Score", 2.99, False, False),
            "drawdown_1y":           ("1Y Max Drawdown", -0.2, False, True),
            "vol_252d":              ("Annual Volatility", 0.25, False, True),
        }

        rows = []
        for key, (label, ref, is_pct, higher_is_worse) in benchmarks.items():
            val = feat_dict.get(key, np.nan)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue

            if higher_is_worse:
                signal = "Increases Risk" if val > ref else "Reduces Risk"
                color = "#dc2626" if val > ref else "#16a34a"
            else:
                signal = "Reduces Risk" if val >= ref else "Increases Risk"
                color = "#16a34a" if val >= ref else "#dc2626"

            rows.append({
                "Feature": label,
                "Value": fmt(val, pct=is_pct),
                "Benchmark": fmt(ref, pct=is_pct),
                "Signal": signal,
                "_color": color,
                "_val": val,
                "_ref": ref,
            })

        df_drivers = pd.DataFrame(rows)

        # Full-width chart
        fig = go.Figure()
        for _, row in df_drivers.iterrows():
            fig.add_trace(go.Bar(
                x=[row["_val"] - row["_ref"]],
                y=[row["Feature"]],
                orientation="h",
                marker=dict(
                    color=row["_color"],
                    line=dict(color="rgba(255,255,255,0.55)", width=1)
                ),
                showlegend=False,
                hovertemplate=(
                    f"{row['Feature']}: {row['Value']}<br>"
                    f"Benchmark: {row['Benchmark']}<br>"
                    f"Signal: {row['Signal']}<extra></extra>"
                )
            ))

        fig.update_layout(
            xaxis_title="Deviation from benchmark",
            height=470,
            margin=dict(t=20, b=20, l=10, r=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            bargap=0.28,
            font={"family": "Inter, Segoe UI, sans-serif", "color": "#475569"}
        )
        fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=True, zerolinecolor="#94a3b8")
        fig.update_yaxes(categoryorder="total ascending")

        st.plotly_chart(fig, use_container_width=True)

        # Cards below chart
        st.markdown("<div class='driver-grid-title'>Benchmark Breakdown</div>", unsafe_allow_html=True)

        card_cols = st.columns(2)
        for i, (_, row) in enumerate(df_drivers.iterrows()):
            with card_cols[i % 2]:
                signal_class = "signal-good" if row["Signal"] == "Reduces Risk" else "signal-bad"
                st.markdown(
                    f"""
                    <div class="driver-card">
                        <div class="driver-card-top">
                            <div class="driver-card-name">{row["Feature"]}</div>
                            <div class="{signal_class}">{row["Signal"]}</div>
                        </div>
                        <div class="driver-card-values">
                            <div>
                                <div class="driver-mini-label">Value</div>
                                <div class="driver-mini-value">{row["Value"]}</div>
                            </div>
                            <div>
                                <div class="driver-mini-label">Benchmark</div>
                                <div class="driver-mini-value">{row["Benchmark"]}</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        increasing = [r["Feature"] for _, r in df_drivers.iterrows() if r["Signal"] == "Increases Risk"]
        reducing   = [r["Feature"] for _, r in df_drivers.iterrows() if r["Signal"] == "Reduces Risk"]

        narrative = ""
        if increasing:
            narrative += f"Risk is negatively impacted by: {', '.join(increasing)}. "
        if reducing:
            narrative += f"Positive signals include: {', '.join(reducing)}."

        st.markdown(f'<div class="highlight-card">{narrative}</div>', unsafe_allow_html=True)

        # Save to session
        st.session_state.update({
            "last_ticker": ticker,
            "last_feat_dict": feat_dict,
            "last_prob": prob,
            "last_altman": altman_z,
            "last_company": company_name,
            "last_hist": hist,
        })

# ==============================================================================
# PAGE 3 — FINANCIAL HEALTH DASHBOARD
# ==============================================================================
elif page == "Financial Health Dashboard":
    render_header(
        "Financial Health Dashboard",
        "A structured view of the company's accounting and market profile",
        "header-dashboard"
    )

    if "last_feat_dict" not in st.session_state:
        st.info("Run a Company Risk Check first to populate this dashboard.")
        st.stop()

    feat_dict    = st.session_state["last_feat_dict"]
    company_name = st.session_state["last_company"]
    hist         = st.session_state["last_hist"]
    prob         = st.session_state["last_prob"]
    risk_label, badge_class, _ = get_risk_label(prob)

    st.markdown(f"""
    <div class="company-strip">
        <div style="font-size:1.45rem; font-weight:800; color:#0f172a;">{company_name}</div>
        <div style="color:#64748b; font-size:0.92rem; margin-top:0.2rem;">
            Current risk classification: <span class="{badge_class}">{risk_label}</span>
            &nbsp;|&nbsp; Distress probability: {prob*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Liquidity")
    l1, l2, l3 = st.columns(3)
    l1.markdown(kpi("Current Ratio", fmt(feat_dict.get("current_ratio"))), unsafe_allow_html=True)
    l2.markdown(kpi("Working Capital", fmt(feat_dict.get("wcap"), dollar=True)), unsafe_allow_html=True)
    l3.markdown(kpi("OCF to Liabilities", fmt(feat_dict.get("ocf_to_liabilities"), pct=True)), unsafe_allow_html=True)

    st.markdown("#### Profitability")
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(kpi("ROA", fmt(feat_dict.get("roa"), pct=True)), unsafe_allow_html=True)
    p2.markdown(kpi("EBIT / Assets", fmt(feat_dict.get("C_ebit_to_assets"), pct=True)), unsafe_allow_html=True)
    p3.markdown(kpi("Net Income", fmt(feat_dict.get("ni"), dollar=True)), unsafe_allow_html=True)
    p4.markdown(kpi("Revenue", fmt(feat_dict.get("sale"), dollar=True)), unsafe_allow_html=True)

    st.markdown("#### Leverage and Solvency")
    v1, v2, v3, v4 = st.columns(4)
    v1.markdown(kpi("Liabilities / Assets", fmt(feat_dict.get("liabilities_to_assets"), pct=True)), unsafe_allow_html=True)
    v2.markdown(kpi("Debt / Assets", fmt(feat_dict.get("debt_to_assets"), pct=True)), unsafe_allow_html=True)
    v3.markdown(kpi("Total Liabilities", fmt(feat_dict.get("lt"), dollar=True)), unsafe_allow_html=True)
    v4.markdown(kpi("Total Assets", fmt(feat_dict.get("at"), dollar=True)), unsafe_allow_html=True)

    st.markdown("#### Market Signals")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(kpi("1M Return", fmt(feat_dict.get("ret_1m"), pct=True)), unsafe_allow_html=True)
    s2.markdown(kpi("12M Return", fmt(feat_dict.get("ret_12m"), pct=True)), unsafe_allow_html=True)
    s3.markdown(kpi("Annual Volatility", fmt(feat_dict.get("vol_252d"), pct=True)), unsafe_allow_html=True)
    s4.markdown(kpi("Market Cap", fmt(feat_dict.get("mkvalt"), dollar=True)), unsafe_allow_html=True)

    if hist is not None and len(hist) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        fig_price = px.line(
            hist.reset_index(), x="Date", y="Close",
            title="12-Month Price History",
            labels={"Close": "Price (USD)", "Date": ""}
        )
        fig_price.update_traces(line_color="#2563eb", line_width=2.5)
        fig_price.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=330,
            margin=dict(t=45, b=20, l=10, r=10),
            font={"family": "Inter, Segoe UI, sans-serif", "color": "#475569"},
            title_font={"size": 15, "color": "#0f172a"}
        )
        fig_price.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
        fig_price.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
        st.plotly_chart(fig_price, use_container_width=True)

# ==============================================================================
# PAGE 4 — METHODOLOGY
# ==============================================================================
elif page == "Methodology":
    render_header(
        "Methodology",
        "How RiskMonitor was built and what drives its predictions",
        "header-method"
    )

    st.markdown("#### Modeling Objective")
    st.markdown("""
    <div class="card">
    The goal is to predict whether a publicly traded company will become financially
    distressed within the next 12 months. Distress is defined using the target variable
    <b>distress_next_1yr_fixed</b>, a binary indicator derived from historical WRDS/Compustat data.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Data Sources")
    st.markdown("""
    <div class="card">
    <b>Training data:</b> WRDS/Compustat — historical accounting statements used to train
    and validate the model. This data is not used at inference time.<br><br>
    <b>Live inference data:</b> Yahoo Finance (via yfinance) — income statement, balance sheet,
    cash flow statement, and 12-month price history pulled at the time of analysis.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Final Model")
    st.markdown("""
    <div class="card">
    <b>HistGradientBoostingClassifier</b> (scikit-learn) was selected as the final model
    after comparison against logistic regression and random forest baselines.
    It handles missing values natively, is robust to outliers with winsorization applied
    during training, and performs well on imbalanced tabular financial data.<br><br>
    Key hyperparameters: learning rate 0.05, max depth 6, 300 iterations,
    min samples per leaf 50, L2 regularization 1.0.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Model Performance")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(kpi("ROC-AUC", f"{float(metrics['roc_auc']):.4f}"), unsafe_allow_html=True)
    mc2.markdown(kpi("Threshold", f"{float(metrics['best_threshold']):.4f}"), unsafe_allow_html=True)
    mc3.markdown(kpi("Training Samples", f"{int(float(metrics['n_train'])):,}"), unsafe_allow_html=True)
    mc4.markdown(kpi("Test Samples", f"{int(float(metrics['n_test'])):,}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Limitations")
    st.markdown("""
    <div class="card">
    <ul style="margin:0; padding-left:1.2rem; line-height:2;">
    <li>yfinance data coverage is incomplete for some smaller or less liquid companies.</li>
    <li>The model was trained on historical distress events and may not generalize to novel macroeconomic environments.</li>
    <li>Predictions are probabilistic estimates, not certainties.</li>
    <li>This tool is for academic and analytical purposes only and does not constitute investment, lending, or legal advice.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
