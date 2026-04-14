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
   render_sidebar_header,
)

st.set_page_config(page_title="Company Risk Check", layout="wide")
apply_global_styles()
render_sidebar_header()

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
        f'<div class="kpi-box"><div class="kpi-label">Altman Z-Score</div><div class="kpi-value">{fmt(altman_z)}<br><span class="{z_badge}" style="font-size:0.72rem;">{z_label}</span></div></div>',
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
            summary = "This company shows elevated next-year distress risk" + (
                f", driven by {', '.join(drivers)}." if drivers else "."
            )
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
        "ocf_to_liabilities":    ("OCF to Liabilities", 0.10, False, False),
        "A_wc_to_assets":        ("Working Capital / Assets", 0.10, False, False),
        "C_ebit_to_assets":      ("EBIT / Assets", 0.08, False, False),
        "Altman_Z":              ("Altman Z-Score", 2.99, False, False),
        "drawdown_1y":           ("1Y Max Drawdown", -0.20, False, True),
        "vol_252d":              ("Annual Volatility", 0.25, False, True),
    }

    rows = []
    for key, (label, ref, is_pct, higher_is_worse) in benchmarks.items():
        val = feat_dict.get(key, np.nan)

        if val is None or pd.isna(val):
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

    if not df_drivers.empty:
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
        reducing = [r["Feature"] for _, r in df_drivers.iterrows() if r["Signal"] == "Reduces Risk"]

        narrative = ""
        if increasing:
            narrative += f"Risk is negatively impacted by: {', '.join(increasing)}. "
        if reducing:
            narrative += f"Positive signals include: {', '.join(reducing)}."

        if narrative.strip():
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
