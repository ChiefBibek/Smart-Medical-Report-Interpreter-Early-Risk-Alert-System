"""
Streamlit Dashboard — Medical Report Interpreter
==================================================
Run with:  streamlit run app.py

Requires: pip install streamlit plotly numpy
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import MedicalRiskPipeline
from preprocessing import REFERENCE_RANGES, FEATURE_NAMES

# ─── Page config ─────────────────────────────
st.set_page_config(
    page_title="Medical Report Interpreter",
    page_icon="🏥",
    layout="wide",
)

DISCLAIMER = (
    "⚠️ **Disclaimer:** This system is for awareness only and is **not** a "
    "replacement for professional medical advice. Always consult a qualified "
    "healthcare professional for diagnosis and treatment."
)

RISK_COLORS = {
    "low":        "#639922",
    "borderline": "#BA7517",
    "moderate":   "#EF9F27",
    "high":       "#E24B4A",
}

# ─── Cached pipeline ─────────────────────────
@st.cache_resource
def load_pipeline():
    pipe = MedicalRiskPipeline()
    pipe.train()
    return pipe


def risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability, 3),
        number={"font": {"size": 32}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1},
            "bar":  {"color": "#378ADD", "thickness": 0.2},
            "steps": [
                {"range": [0.00, 0.20], "color": "#EAF3DE"},
                {"range": [0.20, 0.50], "color": "#FAEEDA"},
                {"range": [0.50, 0.75], "color": "#FAC775"},
                {"range": [0.75, 1.00], "color": "#FCEBEB"},
            ],
            "threshold": {
                "line": {"color": "#E24B4A", "width": 3},
                "thickness": 0.75,
                "value": 0.75
            },
        },
        title={"text": "Risk Probability"}
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def contribution_chart(pct_importance, flags):
    names = list(pct_importance.keys())
    values = list(pct_importance.values())
    colors = ["#E24B4A" if flags.get(n, "normal") != "normal" else "#378ADD" for n in names]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Contributions (%)",
        xaxis_title="Importance (%)",
        height=300,
        margin=dict(l=20, r=60, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis={"categoryorder": "total ascending"},
    )
    return fig


def param_range_chart(raw_values, flags):
    params, patient_vals, normal_mins, normal_maxs, colors = [], [], [], [], []
    for name, val in raw_values.items():
        if name in REFERENCE_RANGES:
            r = REFERENCE_RANGES[name]
            params.append(name)
            patient_vals.append(val)
            normal_mins.append(r["min"])
            normal_maxs.append(r["max"])
            colors.append("#E24B4A" if flags.get(name, "normal") != "normal" else "#639922")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Normal min", x=params, y=normal_mins,
        marker_color="rgba(0,0,0,0)", showlegend=False,
    ))
    fig.add_trace(go.Bar(
        name="Normal range",
        x=params, y=[mx - mn for mx, mn in zip(normal_maxs, normal_mins)],
        base=normal_mins, marker_color="rgba(57,153,34,0.2)",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        name="Patient value", x=params, y=patient_vals,
        mode="markers", marker=dict(color=colors, size=12, symbol="diamond"),
    ))
    fig.update_layout(
        barmode="stack",
        title="Lab Values vs Normal Range",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─── Main app ────────────────────────────────
def main():
    st.title("🏥 Medical Report Interpreter")
    st.caption("AI-based early risk alert system")
    st.warning(DISCLAIMER)
    st.divider()

    pipe = load_pipeline()

    st.sidebar.header("Enter Lab Values")
    st.sidebar.caption("Adjust values or use the demo presets.")

    preset = st.sidebar.selectbox("Quick presets", ["High-Risk Demo", "Normal Demo", "Custom"])

    presets = {
        "High-Risk Demo": dict(hemoglobin=8.2, glucose=210, rbc=3.1, wbc=11.8, platelets=155, creatinine=0.9),
        "Normal Demo":    dict(hemoglobin=14.5, glucose=88, rbc=4.9, wbc=6.5, platelets=250, creatinine=0.8),
        "Custom":         dict(hemoglobin=13.0, glucose=95, rbc=4.7, wbc=7.0, platelets=220, creatinine=0.9),
    }
    defaults = presets[preset]

    inputs = {}
    for name in FEATURE_NAMES:
        r = REFERENCE_RANGES[name]
        inputs[name] = st.sidebar.slider(
            f"{name.capitalize()} ({r['unit']})",
            min_value=float(r["global_min"]),
            max_value=float(r["global_max"]),
            value=float(defaults[name]),
            step=float((r["global_max"] - r["global_min"]) / 100),
        )

    if st.sidebar.button("Analyze", type="primary", use_container_width=True):
        with st.spinner("Running risk assessment..."):
            result = pipe.predict(inputs)
        st.session_state["result"] = result

    if "result" not in st.session_state:
        st.info("👈 Enter lab values in the sidebar and click **Analyze** to see results.")
        return

    r = st.session_state["result"]
    cat = r["risk_category"]
    color = RISK_COLORS[cat]

    # ── Top metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk Score", f"{r['adjusted_probability']:.3f}")
    col2.metric("Category", cat.capitalize())
    col3.metric("Abnormal Flags", sum(1 for v in r["flags"].values() if v != "normal"))
    col4.metric("Override Applied", "Yes ⚠️" if r["override_applied"] else "No ✅")

    st.divider()

    # ── Gauge + contribution ──
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.plotly_chart(risk_gauge(r["adjusted_probability"]), use_container_width=True)
        st.markdown(f"**Confidence:** {r['confidence']}")
    with col_r:
        st.plotly_chart(contribution_chart(r["pct_importance"], r["flags"]), use_container_width=True)

    # ── Parameter chart ──
    st.plotly_chart(param_range_chart(r["raw_values"], r["flags"]), use_container_width=True)

    # ── Explanation ──
    with st.expander("🔍 Model Explanation", expanded=True):
        st.info(r["explanation_text"])
        cols = st.columns(3)
        for i, (name, pct) in enumerate(sorted(r["pct_importance"].items(), key=lambda x: -x[1])[:6]):
            cols[i % 3].metric(name.capitalize(), f"{pct:.1f}%",
                               delta="⚠️ abnormal" if r["flags"].get(name) != "normal" else None)

    # ── Alerts ──
    with st.expander("🚨 Alerts", expanded=True):
        level_style = {"HIGH": "🔴", "MODERATE": "🟡", "BORDERLINE": "⚠️", "LOW": "🟢"}
        for a in r["alerts"][:6]:
            icon = level_style.get(a["level"], "ℹ️")
            st.markdown(f"{icon} **{a['level']}** — {a['message']}")

    # ── Recommendations ──
    with st.expander("💡 Preventive Recommendations", expanded=False):
        for rec in r["recommendations"]:
            st.markdown(f"• {rec['text']}")

    st.caption("This system does not provide diagnoses. For any health concern, consult a qualified doctor.")


if __name__ == "__main__":
    main()
