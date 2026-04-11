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
st.set_page_config(
    page_title="RiskMonitor",
    layout="wide"
)

# ------------------------------------------------------------------------------
# DARK HEADER STYLE
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark top header bar */
    .risk-header {
        background-color: #0f1923;
        padding: 2rem 2.5rem 1.5rem 2.5rem;
        border-radius: 0 0 12px 12px;
        margin-bottom: 2rem;
    }
    .risk-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .risk-header p {
        color: #a0aec0;
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    /* Light content cards */
    .card {
        background-color: #f7f9fc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* KPI metric boxes */
    .kpi-box {
        background-color: #f7f9fc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .kpi-box .kpi-label {
        color: #718096;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .kpi-box .kpi-value {
        color: #1a202c;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    /* Risk badge colors */
    .badge-high {
        background-color: #fff5f5;
        border: 1px solid #fc8181;
        color: #c53030;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-moderate {
        background-color: #fffbeb;
        border: 1px solid #f6ad55;
        color: #c05621;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #f0fff4;
        border: 1px solid #68d391;
        color: #276749;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }

    /* Disclaimer box */
    .disclaimer {
        background-color: #edf2f7;
        border-left: 4px solid #a0aec0;
        border-radius: 6px;
        padding: 0.8rem 1.2rem;
        color: #4a5568;
        font-size: 0.85rem;
        margin-top: 1.5rem;
    }

    /* Section headers */
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #718096;
        margin-bottom: 0.5rem;
    }

    div[data-testid="stSidebar"] {
        background-color: #0f1923;
    }
    div[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
st.sidebar.markdown("## RiskMonitor")
st.sidebar.markdown("---")
page = st.sidebar.radio("", [
    "Home",
    "Company Risk Check",
    "Risk Drivers",
    "Financial Health Dashboard",
    "Methodology"
])

# ------------------------------------------------------------------------------
# HELPER: FETCH AND COMPUTE FEATURES FROM YFINANCE
# ------------------------------------------------------------------------------
def safe_div(a, b):
    try:
        if b and b != 0:
            return a / b
        return np.nan
    except:
        return np.nan

@st.cache_data(show_spinner=False)
def fetch_company_data(ticker: str):
    """Pull financials from yfinance and compute model features."""
    t = yf.Ticker(ticker)

    info   = t.info or {}
    bs     = t.balance_sheet
    inc    = t.income_stmt
    cf     = t.cashflow
    hist   = t.history(period="1y")

    def get(sheet, *rows):
        for row in rows:
            try:
                val = sheet.loc[row].iloc[0]
                if pd.notna(val):
                    return float(val)
            except:
                pass
        return np.nan

    # Core accounting items
    act    = get(bs,  "Current Assets")
    at     = get(bs,  "Total Assets")
    lt     = get(bs,  "Total Liabilities Net Minority Interest", "Total Liabilities")
    lct    = get(bs,  "Current Liabilities")
    wcap   = act - lct if pd.notna(act) and pd.notna(lct) else np.nan
    ebit   = get(inc, "EBIT", "Operating Income")
    ni     = get(inc, "Net Income")
    sale   = get(inc, "Total Revenue")
    revt   = sale
    oancf  = get(cf,  "Operating Cash Flow", "Cash From Operations")
    xint   = get(inc, "Interest Expense")
    re     = get(bs,  "Retained Earnings")
    dltt   = get(bs,  "Long Term Debt")
    dlc    = get(bs,  "Current Debt", "Short Term Debt")

    # Market data
    mkvalt   = info.get("marketCap", np.nan)
    prcc_f   = info.get("previousClose", np.nan)
    last_close = prcc_f

    # Returns and volatility from price history
    if len(hist) > 0:
        closes = hist["Close"]
        ret_1m  = float((closes.iloc[-1] / closes.iloc[max(-21,  -len(closes))] - 1)) if len(closes) >= 5  else np.nan
        ret_3m  = float((closes.iloc[-1] / closes.iloc[max(-63,  -len(closes))] - 1)) if len(closes) >= 20 else np.nan
        ret_6m  = float((closes.iloc[-1] / closes.iloc[max(-126, -len(closes))] - 1)) if len(closes) >= 40 else np.nan
        ret_12m = float((closes.iloc[-1] / closes.iloc[0]))  - 1
        daily_ret = closes.pct_change().dropna()
        vol_30d   = float(daily_ret.iloc[-30:].std()  * np.sqrt(252)) if len(daily_ret) >= 20 else np.nan
        vol_90d   = float(daily_ret.iloc[-90:].std()  * np.sqrt(252)) if len(daily_ret) >= 60 else np.nan
        vol_252d  = float(daily_ret.std() * np.sqrt(252))
        peak      = closes.cummax()
        drawdown_1y = float(((closes - peak) / peak).min())
        avg_volume_30d = float(hist["Volume"].iloc[-30:].mean()) if "Volume" in hist.columns else np.nan
    else:
        ret_1m = ret_3m = ret_6m = ret_12m = np.nan
        vol_30d = vol_90d = vol_252d = np.nan
        drawdown_1y = avg_volume_30d = np.nan

    # Engineered ratios
    liabilities_to_assets = safe_div(lt, at)
    roa                   = safe_div(ni, at)
    ocf_to_liabilities    = safe_div(oancf, lt)
    current_ratio         = safe_div(act, lct)
    debt_to_assets        = safe_div((dltt or 0) + (dlc or 0), at)

    # Altman Z-score components
    A_wc_to_assets   = safe_div(wcap, at)
    B_re_to_assets   = safe_div(re, at)
    C_ebit_to_assets = safe_div(ebit, at)
    D_mve_to_lt      = safe_div(mkvalt, lt)
    E_sales_to_assets= safe_div(sale, at)

    Altman_Z = (
        1.2 * (A_wc_to_assets   or 0) +
        1.4 * (B_re_to_assets   or 0) +
        3.3 * (C_ebit_to_assets or 0) +
        0.6 * (D_mve_to_lt      or 0) +
        1.0 * (E_sales_to_assets or 0)
    )

    feature_dict = {
        "Altman_Z":            Altman_Z,
        "A_wc_to_assets":      A_wc_to_assets,
        "B_re_to_assets":      B_re_to_assets,
        "C_ebit_to_assets":    C_ebit_to_assets,
        "D_mve_to_lt":         D_mve_to_lt,
        "E_sales_to_assets":   E_sales_to_assets,
        "act":                 act,
        "at":                  at,
        "lt":                  lt,
        "wcap":                wcap,
        "ebit":                ebit,
        "ni":                  ni,
        "sale":                sale,
        "revt":                revt,
        "oancf":               oancf,
        "xint":                xint,
        "mkvalt":              mkvalt,
        "prcc_f":              prcc_f,
        "last_close":          last_close,
        "ret_1m":              ret_1m,
        "ret_3m":              ret_3m,
        "ret_6m":              ret_6m,
        "ret_12m":             ret_12m,
        "vol_30d":             vol_30d,
        "vol_90d":             vol_90d,
        "vol_252d":            vol_252d,
        "drawdown_1y":         drawdown_1y,
        "avg_volume_30d":      avg_volume_30d,
        "liabilities_to_assets": liabilities_to_assets,
        "roa":                 roa,
        "ocf_to_liabilities":  ocf_to_liabilities,
        "current_ratio":       current_ratio,
        "debt_to_assets":      debt_to_assets,
    }

    company_name = info.get("longName", ticker.upper())
    sector       = info.get("sector", "N/A")
    industry     = info.get("industry", "N/A")

    return feature_dict, company_name, sector, industry, Altman_Z, info, hist

def get_risk_label(prob):
    if prob >= THRESHOLD * 1.3:
        return "High Risk", "badge-high"
    elif prob >= THRESHOLD * 0.8:
        return "Moderate Risk", "badge-moderate"
    else:
        return "Low Risk", "badge-low"

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
        elif abs(val) >= 1e6:
            return f"${val/1e6:.1f}M"
        return f"${val:,.0f}"
    if pct:
        return f"{val*100:.1f}%"
    return f"{val:.{decimals}f}"

# ==============================================================================
# PAGE 1 — HOME
# ==============================================================================
if page == "Home":
    st.markdown("""
    <div class="risk-header">
        <h1>RiskMonitor</h1>
        <p>Early-warning financial distress prediction for public companies</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### What this app does")
    st.markdown("""
    <div class="card">
    RiskMonitor estimates the probability that a publicly traded company will become
    financially distressed within the next 12 months. It combines a machine learning model
    trained on historical accounting data with live financial and market signals pulled
    directly from public sources. The result is an interpretable risk score grounded in
    both classical finance theory and modern predictive modeling.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Why it matters")
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

    st.markdown("#### How to use it")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="kpi-box">
            <div class="kpi-label">Step 1</div>
            <div class="kpi-value" style="font-size:1rem; margin-top:0.5rem;">Enter a ticker on the Company Risk Check page</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="kpi-box">
            <div class="kpi-label">Step 2</div>
            <div class="kpi-value" style="font-size:1rem; margin-top:0.5rem;">Review the predicted risk score and explanation</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="kpi-box">
            <div class="kpi-label">Step 3</div>
            <div class="kpi-value" style="font-size:1rem; margin-top:0.5rem;">Explore financial health and risk drivers in depth</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Model highlights")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown('<div class="kpi-box"><div class="kpi-label">Model</div><div class="kpi-value" style="font-size:1.1rem;">HistGradientBoosting</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="kpi-box"><div class="kpi-label">ROC-AUC</div><div class="kpi-value">{float(metrics["roc_auc"]):.4f}</div></div>', unsafe_allow_html=True)
    m3.markdown('<div class="kpi-box"><div class="kpi-label">Target</div><div class="kpi-value" style="font-size:1.1rem;">Next-Year Distress</div></div>', unsafe_allow_html=True)
    m4.markdown('<div class="kpi-box"><div class="kpi-label">Benchmark</div><div class="kpi-value" style="font-size:1.1rem;">Altman Z-Score</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
    This tool is for academic and analytical purposes only. It is not investment, lending, or legal advice.
    Predictions are probabilistic and based on publicly available data that may be incomplete or delayed.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 2 — COMPANY RISK CHECK
# ==============================================================================
elif page == "Company Risk Check":
    st.markdown("""
    <div class="risk-header">
        <h1>Company Risk Check</h1>
        <p>Enter a ticker to generate an instant distress risk score</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, _ = st.columns([1, 3])
    with col_input:
        ticker = st.text_input("Stock Ticker", placeholder="e.g. AAPL, F, GME").strip().upper()
        run    = st.button("Analyze", use_container_width=True)

    if run and ticker:
        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                feat_dict, company_name, sector, industry, altman_z, info, hist = fetch_company_data(ticker)
            except Exception as e:
                st.error(f"Could not retrieve data for {ticker}. Check the ticker and try again.")
                st.stop()

        X    = pd.DataFrame([feat_dict])[features]
        prob = float(model.predict_proba(X)[0][1])
        risk_label, badge_class = get_risk_label(prob)
        z_label, z_badge        = get_altman_label(altman_z)

        st.markdown(f"### {company_name}")
        st.markdown(f"**{sector}** — {industry}")
        st.divider()

        # Top KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'<div class="kpi-box"><div class="kpi-label">Distress Probability</div><div class="kpi-value">{prob*100:.1f}%</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi-box"><div class="kpi-label">Risk Category</div><div class="kpi-value" style="font-size:1.1rem;"><span class="{badge_class}">{risk_label}</span></div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi-box"><div class="kpi-label">Model Threshold</div><div class="kpi-value">{THRESHOLD:.4f}</div></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi-box"><div class="kpi-label">Altman Z-Score</div><div class="kpi-value">{altman_z:.2f} <span class="{z_badge}" style="font-size:0.75rem;">{z_label}</span></div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge chart
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": "#e53e3e" if prob >= THRESHOLD else "#48bb78"},
                "steps": [
                    {"range": [0,   THRESHOLD*80],        "color": "#f0fff4"},
                    {"range": [THRESHOLD*80, THRESHOLD*130], "color": "#fffbeb"},
                    {"range": [THRESHOLD*130, 100],        "color": "#fff5f5"},
                ],
                "threshold": {
                    "line":  {"color": "#2d3748", "width": 3},
                    "thickness": 0.75,
                    "value": THRESHOLD * 100
                }
            },
            title={"text": "Distress Probability", "font": {"size": 16}}
        ))
        gauge.update_layout(height=280, margin=dict(t=40, b=10, l=30, r=30))

        g_col, s_col = st.columns([1, 1])
        with g_col:
            st.plotly_chart(gauge, use_container_width=True)
        with s_col:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("**Plain-English Summary**")

            drivers = []
            if feat_dict.get("roa", 0) and feat_dict["roa"] < 0:
                drivers.append("negative profitability (ROA below zero)")
            if feat_dict.get("liabilities_to_assets", 0) and feat_dict["liabilities_to_assets"] > 0.7:
                drivers.append("high leverage (liabilities over 70% of assets)")
            if feat_dict.get("current_ratio", 2) and feat_dict["current_ratio"] < 1:
                drivers.append("weak liquidity (current ratio below 1)")
            if feat_dict.get("ocf_to_liabilities", 0) and feat_dict["ocf_to_liabilities"] < 0:
                drivers.append("negative operating cash flow relative to liabilities")
            if altman_z < 1.81:
                drivers.append("an Altman Z-score in the distress zone")

            if prob >= THRESHOLD:
                if drivers:
                    summary = f"This company shows elevated next-year distress risk, driven by {', '.join(drivers)}."
                else:
                    summary = "This company shows elevated next-year distress risk based on the combined model signal."
            else:
                summary = "Based on current financial and market data, this company does not show elevated distress risk relative to the model threshold."

            st.markdown(f'<div class="card">{summary}</div>', unsafe_allow_html=True)

        # Benchmark strip
        st.markdown("#### Benchmark Signals")
        b1, b2, b3, b4 = st.columns(4)
        b1.markdown(f'<div class="kpi-box"><div class="kpi-label">Altman Z-Score</div><div class="kpi-value">{fmt(altman_z)}</div></div>', unsafe_allow_html=True)
        b2.markdown(f'<div class="kpi-box"><div class="kpi-label">Current Ratio</div><div class="kpi-value">{fmt(feat_dict.get("current_ratio"))}</div></div>', unsafe_allow_html=True)
        b3.markdown(f'<div class="kpi-box"><div class="kpi-label">ROA</div><div class="kpi-value">{fmt(feat_dict.get("roa"), pct=True)}</div></div>', unsafe_allow_html=True)
        b4.markdown(f'<div class="kpi-box"><div class="kpi-label">Debt to Assets</div><div class="kpi-value">{fmt(feat_dict.get("debt_to_assets"), pct=True)}</div></div>', unsafe_allow_html=True)

        st.session_state["last_ticker"]    = ticker
        st.session_state["last_feat_dict"] = feat_dict
        st.session_state["last_prob"]      = prob
        st.session_state["last_altman"]    = altman_z
        st.session_state["last_company"]   = company_name
        st.session_state["last_hist"]      = hist

# ==============================================================================
# PAGE 3 — RISK DRIVERS
# ==============================================================================
elif page == "Risk Drivers":
    st.markdown("""
    <div class="risk-header">
        <h1>Risk Drivers</h1>
        <p>Understand what is driving the model's prediction</p>
    </div>
    """, unsafe_allow_html=True)

    if "last_feat_dict" not in st.session_state:
        st.info("Run a Company Risk Check first to see risk drivers.")
        st.stop()

    feat_dict    = st.session_state["last_feat_dict"]
    prob         = st.session_state["last_prob"]
    company_name = st.session_state["last_company"]

    st.markdown(f"### {company_name} — Risk Driver Analysis")

    # Reference medians (approximate healthy benchmarks)
    benchmarks = {
        "roa":                  ("ROA",                     0.05,  True,  False),
        "current_ratio":        ("Current Ratio",            1.5,   False, False),
        "liabilities_to_assets":("Liabilities to Assets",   0.5,   False, True),
        "debt_to_assets":       ("Debt to Assets",           0.35,  False, True),
        "ocf_to_liabilities":   ("OCF to Liabilities",       0.1,   False, False),
        "A_wc_to_assets":       ("Working Capital / Assets", 0.1,   False, False),
        "C_ebit_to_assets":     ("EBIT / Assets",            0.08,  False, False),
        "Altman_Z":             ("Altman Z-Score",           2.99,  False, False),
        "drawdown_1y":          ("1Y Max Drawdown",         -0.2,   False, True),
        "vol_252d":             ("Annual Volatility",        0.25,  False, True),
    }

    rows = []
    for key, (label, ref, is_pct, higher_is_worse) in benchmarks.items():
        val = feat_dict.get(key, np.nan)
        if val is None or np.isnan(val):
            continue
        gap = val - ref
        if higher_is_worse:
            signal = "Increases Risk" if val > ref else "Reduces Risk"
            color  = "#c53030" if val > ref else "#276749"
        else:
            signal = "Reduces Risk" if val >= ref else "Increases Risk"
            color  = "#276749" if val >= ref else "#c53030"

        rows.append({
            "Feature":    label,
            "Value":      fmt(val, pct=is_pct),
            "Benchmark":  fmt(ref, pct=is_pct),
            "Signal":     signal,
            "_color":     color,
            "_val":       val,
            "_ref":       ref,
            "_worse":     higher_is_worse,
        })

    df_drivers = pd.DataFrame(rows)

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("**Feature vs Benchmark**")
        fig = go.Figure()
        for _, row in df_drivers.iterrows():
            color = row["_color"]
            fig.add_trace(go.Bar(
                x=[row["_val"] - row["_ref"]],
                y=[row["Feature"]],
                orientation="h",
                marker_color=color,
                showlegend=False,
                hovertemplate=f"{row['Feature']}: {row['Value']} (benchmark {row['Benchmark']})<extra></extra>"
            ))
        fig.update_layout(
            xaxis_title="Deviation from benchmark",
            height=380,
            margin=dict(t=20, b=20, l=10, r=10),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**Signal Breakdown**")
        for _, row in df_drivers.iterrows():
            color = row["_color"]
            st.markdown(
                f'<div class="card" style="border-left: 4px solid {color}; padding: 0.6rem 1rem;">'
                f'<b>{row["Feature"]}</b><br>'
                f'<span style="color:#4a5568;">Value: {row["Value"]} &nbsp;|&nbsp; Benchmark: {row["Benchmark"]}</span><br>'
                f'<span style="color:{color}; font-weight:600;">{row["Signal"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    increasing = [r["Feature"] for _, r in df_drivers.iterrows() if r["Signal"] == "Increases Risk"]
    reducing   = [r["Feature"] for _, r in df_drivers.iterrows() if r["Signal"] == "Reduces Risk"]
    narrative  = ""
    if increasing:
        narrative += f"The company's risk profile is negatively impacted by: {', '.join(increasing)}. "
    if reducing:
        narrative += f"Positive signals include: {', '.join(reducing)}."
    st.markdown(f'<div class="card">{narrative}</div>', unsafe_allow_html=True)

# ==============================================================================
# PAGE 4 — FINANCIAL HEALTH DASHBOARD
# ==============================================================================
elif page == "Financial Health Dashboard":
    st.markdown("""
    <div class="risk-header">
        <h1>Financial Health Dashboard</h1>
        <p>A structured view of the company's accounting and market profile</p>
    </div>
    """, unsafe_allow_html=True)

    if "last_feat_dict" not in st.session_state:
        st.info("Run a Company Risk Check first to populate this dashboard.")
        st.stop()

    feat_dict    = st.session_state["last_feat_dict"]
    company_name = st.session_state["last_company"]
    hist         = st.session_state["last_hist"]

    st.markdown(f"### {company_name}")

    # Liquidity
    st.markdown("#### Liquidity")
    l1, l2, l3 = st.columns(3)
    l1.markdown(f'<div class="kpi-box"><div class="kpi-label">Current Ratio</div><div class="kpi-value">{fmt(feat_dict.get("current_ratio"))}</div></div>', unsafe_allow_html=True)
    l2.markdown(f'<div class="kpi-box"><div class="kpi-label">Working Capital</div><div class="kpi-value">{fmt(feat_dict.get("wcap"), dollar=True)}</div></div>', unsafe_allow_html=True)
    l3.markdown(f'<div class="kpi-box"><div class="kpi-label">OCF to Liabilities</div><div class="kpi-value">{fmt(feat_dict.get("ocf_to_liabilities"), pct=True)}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Profitability
    st.markdown("#### Profitability")
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(f'<div class="kpi-box"><div class="kpi-label">ROA</div><div class="kpi-value">{fmt(feat_dict.get("roa"), pct=True)}</div></div>', unsafe_allow_html=True)
    p2.markdown(f'<div class="kpi-box"><div class="kpi-label">EBIT / Assets</div><div class="kpi-value">{fmt(feat_dict.get("C_ebit_to_assets"), pct=True)}</div></div>', unsafe_allow_html=True)
    p3.markdown(f'<div class="kpi-box"><div class="kpi-label">Net Income</div><div class="kpi-value">{fmt(feat_dict.get("ni"), dollar=True)}</div></div>', unsafe_allow_html=True)
    p4.markdown(f'<div class="kpi-box"><div class="kpi-label">Revenue</div><div class="kpi-value">{fmt(feat_dict.get("sale"), dollar=True)}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Leverage
    st.markdown("#### Leverage and Solvency")
    v1, v2, v3, v4 = st.columns(4)
    v1.markdown(f'<div class="kpi-box"><div class="kpi-label">Liabilities / Assets</div><div class="kpi-value">{fmt(feat_dict.get("liabilities_to_assets"), pct=True)}</div></div>', unsafe_allow_html=True)
    v2.markdown(f'<div class="kpi-box"><div class="kpi-label">Debt / Assets</div><div class="kpi-value">{fmt(feat_dict.get("debt_to_assets"), pct=True)}</div></div>', unsafe_allow_html=True)
    v3.markdown(f'<div class="kpi-box"><div class="kpi-label">Total Liabilities</div><div class="kpi-value">{fmt(feat_dict.get("lt"), dollar=True)}</div></div>', unsafe_allow_html=True)
    v4.markdown(f'<div class="kpi-box"><div class="kpi-label">Total Assets</div><div class="kpi-value">{fmt(feat_dict.get("at"), dollar=True)}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Market signals + price chart
    st.markdown("#### Market Signals")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(f'<div class="kpi-box"><div class="kpi-label">1M Return</div><div class="kpi-value">{fmt(feat_dict.get("ret_1m"), pct=True)}</div></div>', unsafe_allow_html=True)
    s2.markdown(f'<div class="kpi-box"><div class="kpi-label">12M Return</div><div class="kpi-value">{fmt(feat_dict.get("ret_12m"), pct=True)}</div></div>', unsafe_allow_html=True)
    s3.markdown(f'<div class="kpi-box"><div class="kpi-label">Annual Volatility</div><div class="kpi-value">{fmt(feat_dict.get("vol_252d"), pct=True)}</div></div>', unsafe_allow_html=True)
    s4.markdown(f'<div class="kpi-box"><div class="kpi-label">Market Cap</div><div class="kpi-value">{fmt(feat_dict.get("mkvalt"), dollar=True)}</div></div>', unsafe_allow_html=True)

    if hist is not None and len(hist) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        fig_price = px.line(
            hist.reset_index(),
            x="Date",
            y="Close",
            title="12-Month Price History",
            labels={"Close": "Price (USD)", "Date": ""}
        )
        fig_price.update_traces(line_color="#2b6cb0", line_width=2)
        fig_price.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=300,
            margin=dict(t=40, b=20, l=10, r=10)
        )
        st.plotly_chart(fig_price, use_container_width=True)

# ==============================================================================
# PAGE 5 — METHODOLOGY
# ==============================================================================
elif page == "Methodology":
    st.markdown("""
    <div class="risk-header">
        <h1>Methodology</h1>
        <p>How RiskMonitor was built and what drives its predictions</p>
    </div>
    """, unsafe_allow_html=True)

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
    during training, and performs well on imbalanced tabular financial data.
    <br><br>
    Key hyperparameters: learning rate 0.05, max depth 6, 300 iterations,
    min samples per leaf 50, L2 regularization 1.0.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Model Performance")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(f'<div class="kpi-box"><div class="kpi-label">ROC-AUC</div><div class="kpi-value">{float(metrics["roc_auc"]):.4f}</div></div>', unsafe_allow_html=True)
    mc2.markdown(f'<div class="kpi-box"><div class="kpi-label">Threshold</div><div class="kpi-value">{float(metrics["best_threshold"]):.4f}</div></div>', unsafe_allow_html=True)
    mc3.markdown(f'<div class="kpi-box"><div class="kpi-label">Training Samples</div><div class="kpi-value">{int(float(metrics["n_train"])):,}</div></div>', unsafe_allow_html=True)
    mc4.markdown(f'<div class="kpi-box"><div class="kpi-label">Test Samples</div><div class="kpi-value">{int(float(metrics["n_test"])):,}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Limitations")
    st.markdown("""
    <div class="card">
    <ul>
    <li>yfinance data coverage is incomplete for some smaller or less liquid companies.</li>
    <li>The model was trained on historical distress events and may not generalize to
    novel macroeconomic environments.</li>
    <li>Predictions are probabilistic estimates, not certainties.</li>
    <li>This tool is for academic and analytical purposes only and does not constitute
    investment, lending, or legal advice.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
