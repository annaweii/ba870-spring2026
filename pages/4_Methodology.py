import streamlit as st
from utils import load_css, render_header, kpi, metrics

st.set_page_config(page_title="Methodology", layout="wide")
load_css()

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
