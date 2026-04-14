import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import yfinance as yf

@st.cache_resource
def load_artifacts():
    model = joblib.load("final_app_model.joblib")

    with open("final_app_features.json", "r") as f:
        features = json.load(f)

    with open("final_app_threshold.json", "r") as f:
        thresh_cfg = json.load(f)

    threshold = thresh_cfg["threshold"]
    metrics = pd.read_csv("final_model_metrics.csv").set_index("metric")["value"]

    return model, features, threshold, metrics

def apply_global_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            background-color: #f4f7fb;
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #061225 0%, #0b1f45 100%);
            border-right: 1px solid #18345f;
        }
        div[data-testid="stSidebar"] * {
            color: #d9e3f0 !important;
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

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

        .kpi-box {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.15rem 1rem;
            text-align: center;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
            height: 100%;
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
            white-space: nowrap;
        }

        .kpi-box-blue {
            background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%);
            border: 1px solid #bfdbfe;
        }
        .kpi-box-teal {
            background: linear-gradient(180deg, #f0fdfa 0%, #ffffff 100%);
            border: 1px solid #99f6e4;
        }
        .kpi-box-purple {
            background: linear-gradient(180deg, #f5f3ff 0%, #ffffff 100%);
            border: 1px solid #ddd6fe;
        }
        .kpi-box-slate {
            background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
            border: 1px solid #cbd5e1;
        }
        .kpi-box-amber {
            background: linear-gradient(180deg, #fffbeb 0%, #ffffff 100%);
            border: 1px solid #fde68a;
        }

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

        .stButton > button {
            background: linear-gradient(135deg, #1d4ed8, #2563eb, #3b82f6);
            color: #ffffff;
            border: none;
            border-radius: 10px;
            font-weight: 700;
            font-size: 0.95rem;
            padding: 0.55rem 1.3rem;
            box-shadow: 0 6px 16px rgba(37,99,235,0.28);
        }

        .stTextInput input {
            border-radius: 10px;
            border: 1.5px solid #cbd5e1;
            font-size: 1rem;
            padding: 0.55rem 1rem;
            background: #ffffff;
        }

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

def safe_div(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return a / b
    except Exception:
        return np.nan

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

def kpi(label, value, box_class=""):
    return f'<div class="kpi-box {box_class}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'

def get_risk_label(prob, threshold):
    if prob >= threshold * 1.3:
        return "High Risk", "badge-high", "banner-high"
    elif prob >= threshold * 0.8:
        return "Moderate Risk", "badge-moderate", "banner-moderate"
    return "Low Risk", "badge-low", "banner-low"

def get_altman_label(z):
    if pd.isna(z):
        return "N/A", "badge-moderate"
    if z < 1.81:
        return "Distress Zone", "badge-high"
    elif z < 2.99:
        return "Grey Zone", "badge-moderate"
    return "Safe Zone", "badge-low"

@st.cache_data(show_spinner=False)
def fetch_company_data(ticker: str):
    t = yf.Ticker(ticker)

    try:
        info = t.info or {}
    except Exception:
        info = {}

    try:
        bs = t.balance_sheet
    except Exception:
        bs = pd.DataFrame()

    try:
        inc = t.income_stmt
    except Exception:
        inc = pd.DataFrame()

    try:
        cf = t.cashflow
    except Exception:
        cf = pd.DataFrame()

    try:
        hist = t.history(period="1y")
    except Exception:
        hist = pd.DataFrame()

    if hist is None or hist.empty:
        raise ValueError("No price history returned from Yahoo Finance.")

    def get(sheet, *rows):
        if sheet is None or sheet.empty:
            return np.nan
        for row in rows:
            try:
                val = sheet.loc[row].iloc[0]
                if pd.notna(val):
                    return float(val)
            except Exception:
                pass
        return np.nan

    act = get(bs, "Current Assets")
    at = get(bs, "Total Assets")
    lt = get(bs, "Total Liabilities Net Minority Interest", "Total Liabilities")
    lct = get(bs, "Current Liabilities")
    wcap = act - lct if pd.notna(act) and pd.notna(lct) else np.nan
    ebit = get(inc, "EBIT", "Operating Income")
    ni = get(inc, "Net Income")
    sale = get(inc, "Total Revenue")
    oancf = get(cf, "Operating Cash Flow", "Cash From Operations")
    xint = get(inc, "Interest Expense")
    re = get(bs, "Retained Earnings")
    dltt = get(bs, "Long Term Debt")
    dlc = get(bs, "Current Debt", "Short Term Debt")

    mkvalt = info.get("marketCap", np.nan)
    prcc_f = info.get("previousClose", np.nan)
    last_close = prcc_f

    closes = hist["Close"]
    ret_1m = float(closes.iloc[-1] / closes.iloc[-21] - 1) if len(closes) >= 21 else np.nan
    ret_3m = float(closes.iloc[-1] / closes.iloc[-63] - 1) if len(closes) >= 63 else np.nan
    ret_6m = float(closes.iloc[-1] / closes.iloc[-126] - 1) if len(closes) >= 126 else np.nan
    ret_12m = float(closes.iloc[-1] / closes.iloc[0] - 1) if len(closes) > 1 else np.nan

    daily_ret = closes.pct_change().dropna()
    vol_30d = float(daily_ret.iloc[-30:].std() * np.sqrt(252)) if len(daily_ret) >= 30 else np.nan
    vol_90d = float(daily_ret.iloc[-90:].std() * np.sqrt(252)) if len(daily_ret) >= 90 else np.nan
    vol_252d = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 0 else np.nan
    peak = closes.cummax()
    drawdown_1y = float(((closes - peak) / peak).min()) if len(closes) > 0 else np.nan
    avg_volume_30d = float(hist["Volume"].iloc[-30:].mean()) if "Volume" in hist.columns and len(hist) >= 30 else np.nan

    liabilities_to_assets = safe_div(lt, at)
    roa = safe_div(ni, at)
    ocf_to_liabilities = safe_div(oancf, lt)
    current_ratio = safe_div(act, lct)
    debt_total = (0 if pd.isna(dltt) else dltt) + (0 if pd.isna(dlc) else dlc)
    debt_to_assets = safe_div(debt_total, at)

    A_wc_to_assets = safe_div(wcap, at)
    B_re_to_assets = safe_div(re, at)
    C_ebit_to_assets = safe_div(ebit, at)
    D_mve_to_lt = safe_div(mkvalt, lt)
    E_sales_to_assets = safe_div(sale, at)

    Altman_Z = (
        1.2 * (0 if pd.isna(A_wc_to_assets) else A_wc_to_assets) +
        1.4 * (0 if pd.isna(B_re_to_assets) else B_re_to_assets) +
        3.3 * (0 if pd.isna(C_ebit_to_assets) else C_ebit_to_assets) +
        0.6 * (0 if pd.isna(D_mve_to_lt) else D_mve_to_lt) +
        1.0 * (0 if pd.isna(E_sales_to_assets) else E_sales_to_assets)
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
