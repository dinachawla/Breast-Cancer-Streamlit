import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path
from sklearn.datasets import load_breast_cancer

st.set_page_config(layout="wide")
MODEL_PATH = Path("breast_cancer_pipe_updated.pkl")
TEST_ACC = 0.965

# Load data + model
@st.cache_resource
def load_model(path):
    return joblib.load(path)

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
pipe = load_model(MODEL_PATH)

# Use only the 11 features
FEATURE_GROUPS = {
    "Radius (mm)": ["mean radius", "worst radius"],
    "Perimeter (mm)": ["mean perimeter", "worst perimeter"],
    "Area (mm²)": ["mean area", "worst area"],
    "Concavity": ["mean concavity", "worst concavity"],
    "Concave Points": ["mean concave points", "worst concave points"],
    "Texture": ["mean texture"]
}
USED_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

# Prepare slider ranges
percentile_bounds = {}
for col in USED_FEATURES:
    low = np.percentile(df[col], 5)
    high = np.percentile(df[col], 95)
    avg = df[col].mean()
    step = 0.01 if high - low < 10 else 0.1 if high - low < 100 else 1
    percentile_bounds[col] = (low, high, step, avg)

# Session state init
for col in USED_FEATURES:
    if f"s_{col}" not in st.session_state:
        st.session_state[f"s_{col}"] = percentile_bounds[col][3]

# Layout
st.title("Breast Cancer ML Classifier 🩺")
st.caption(f"Model hold-out accuracy: {TEST_ACC:.1%}")
st.subheader("Adjust Tumor Characteristics")

left, right = st.columns([1, 1], gap="large")

with left:
    values = {}
    for group, features in FEATURE_GROUPS.items():
        st.markdown(f"### {group}")
        cols = st.columns(len(features))
        for col, feat in zip(cols, features):
            low, high, step, avg = percentile_bounds[feat]
            with col:
                st.markdown(f"<h4 style='margin-bottom:0.2rem'>{feat.title()}</h4>", unsafe_allow_html=True)
                st.caption(f"*Population average: {avg:.3f}*")
                val = st.slider("", min_value=float(low), max_value=float(high), step=float(step),
                                value=st.session_state[f"s_{feat}"], key=f"s_{feat}",
                                label_visibility="collapsed")
                values[feat] = val

with right:
    st.markdown("""
    <div style="position:sticky; top:0; background-color:#0e1117; padding:1rem 1rem 0 1rem; z-index:99">
    <h2 style="margin-bottom:0.5rem">Feature-Level Malignancy Likelihood</h2>
    <p style="margin-top:0; font-weight:500">Estimated Malignancy Likelihood per Feature</p>
    """, unsafe_allow_html=True)

    chart_data = []
    for feat, val in values.items():
        delta = 0.05 * val
        sub_df = df[(df[feat] >= val - delta) & (df[feat] <= val + delta)]
        malignant_pct = 100 * (1 - sub_df["target"].mean()) if not sub_df.empty else None
        chart_data.append((feat, malignant_pct))

    filtered_df = pd.DataFrame(chart_data, columns=["Feature", "% Malignant"]).dropna()
    if not filtered_df.empty:
        fig = go.Figure(go.Scatter(
            x=filtered_df["Feature"],
            y=filtered_df["% Malignant"],
            mode='lines+markers+text',
            text=[f"{p:.1f}%" for p in filtered_df["% Malignant"]],
            textposition="top center",
            line=dict(color="crimson", width=3)
        ))
        fig.update_layout(
            xaxis_title="Tumor Feature",
            yaxis_title="% of Similar Cases that were Malignant",
            yaxis_range=[0, 100],
            height=500,
            margin=dict(l=10, r=10, t=10, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to show chart.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Diagnosis Estimate")

    # Real-time prediction
    ordered_values = np.array([[values[f] for f in USED_FEATURES]])
    pred_prob = pipe.predict_proba(ordered_values)[0][1]  # Probability of MALIGNANT
    pred_class = "MALIGNANT" if pred_prob >= 0.5 else "BENIGN"

    if pred_class == "MALIGNANT":
        st.error(f"🚨 **MALIGNANT**  \nProbability: **{pred_prob:.1%}** (≈ {pred_prob*100:.0f} out of 100 similar cases)", icon="🚨")
    else:
        st.success(f"🫰 **BENIGN**  \nProbability: **{1 - pred_prob:.1%}** (≈ {(1 - pred_prob)*100:.0f} out of 100 similar cases)", icon="✅")

    st.caption("Model is for educational use only and **does not replace professional medical advice.**")
