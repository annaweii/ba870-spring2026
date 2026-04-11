import streamlit as st
import plotly.express as px

from utils import load_css, render_header, get_risk_label, kpi, fmt

st.set_page_config(page_title="Financial Health Dashboard", layout="wide")
load_css()

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
