import streamlit as st
from utils import load_css

st.set_page_config(page_title="RiskMonitor", layout="wide")
load_css()

st.sidebar.markdown("## RiskMonitor")
st.sidebar.markdown("Use the page menu to explore the app.")

st.markdown("""
<div class="page-header header-home">
    <h1>RiskMonitor</h1>
    <p>Early-warning financial distress prediction for public companies</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
This app uses a machine learning model with live financial and market data to estimate
the probability of next-year financial distress for public companies.
Use the page navigation on the left to move between Home, Risk Check, Dashboard, and Methodology.
</div>
""", unsafe_allow_html=True)
