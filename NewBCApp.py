import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path
from sklearn.datasets import load_breast_cancer

st.set_page_config(layout="wide")

# Sticky styling + dark-mode metric boxes
st.markdown("""
    <style>
    [data-testid="column"]:nth-of-type(3) > div {
        position: sticky;
        top: 80px;
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        z-index: 2;
    }
    .metric-box {
        background-color: #2a2a2a;
        color: #e1e1e1;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-box p {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("breast_cancer_pipe_11features.pkl")
TEST_ACC = 0.965

# Load data and model
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['is_malignant'] = (data.target == 0).astype(int)

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

pipe = load_model(MODEL_PATH)

# Pull out the final estimator’s coefs
model = pipe.steps[-1][1]
raw_coefs = getattr(model, 'coef_', [[0]*len(pipe.feature_names_in_)])[0]

SELECTED_FEATURES = [
    "mean radius", "worst radius",
    "mean perimeter", "worst perimeter",
    "mean area", "worst area",
    "mean concavity", "worst concavity",
    "mean concave points", "worst concave points",
    "mean texture"
]
feature_coefs = {feat: float(raw_coefs[i]) for i, feat in enumerate(SELECTED_FEATURES)}

# Percentile bounds
percentile_bounds = {}
for feat in SELECTED_FEATURES:
    low, high = np.percentile(df[feat], [5, 95])
    avg = df[feat].mean()
    span = high - low
    step = 0.001 if span < 10 else 0.1 if span < 100 else 1
    percentile_bounds[feat] = (low, high, step, avg)

FEATURE_GROUPS = {
    "Radius (mm)": ["mean radius", "worst radius"],
    "Perimeter (mm)": ["mean perimeter", "worst perimeter"],
    "Area (mm²)": ["mean area", "worst area"],
    "Concavity": ["mean concavity", "worst concavity"],
    "Concave Points": ["mean concave points", "worst concave points"],
    "Texture": ["mean texture"]
}

# Session-state init
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = None

for feat in SELECTED_FEATURES:
    if f"s_{feat}" not in st.session_state:
        st.session_state[f"s_{feat}"] = percentile_bounds[feat][3]
        st.session_state[f"n_{feat}"] = percentile_bounds[feat][3]

if st.session_state.reset_trigger:
    rk = st.session_state.reset_trigger
    _, _, _, avg = percentile_bounds[rk]
    st.session_state[f"s_{rk}"] = avg
    st.session_state[f"n_{rk}"] = avg
    st.session_state.reset_trigger = None

def sync_slider(f): st.session_state[f"s_{f}"] = st.session_state[f"n_{f}"]
def sync_number(f): st.session_state[f"n_{f}"] = st.session_state[f"s_{f}"]

# Build UI
st.title("Breast Cancer ML Classifier 🩺")
st.caption(f"Model hold-out accuracy: {TEST_ACC:.1%}")
st.subheader("Adjust Tumor Characteristics")

col_left, col_right = st.columns([1,1], gap="large")

with col_left:
    values = {}
    for group, feats in FEATURE_GROUPS.items():
        st.markdown(f"### {group}")
        cols = st.columns(len(feats))
        for container, feat in zip(cols, feats):
            with container:
                coef = feature_coefs.get(feat, 0.0)
                direction_word = "Increasing" if coef > 0 else "Decreasing"
                low, high, step, avg = percentile_bounds[feat]

                # — Subheading + direction on one line —
                st.markdown(
                    f"""
<div class='metric-box'>
  <p style='margin-bottom:0.25rem;'>
    <strong style='display:inline-block; font-size:1rem'>{feat.title()}</strong>
    &nbsp;<strong style='display:inline'>{direction_word}</strong> {feat} indicates malignancy.
  </p>
  <p><em>Population average: {avg:.3f}</em></p>
</div>
""",
                    unsafe_allow_html=True
                )

                st.slider(
                    label="", key=f"s_{feat}",
                    min_value=float(low), max_value=float(high),
                    step=float(step), label_visibility="collapsed",
                    on_change=sync_number, args=(feat,)
                )
                st.number_input(
                    "Exact", key=f"n_{feat}",
                    min_value=float(low), max_value=float(high),
                    step=float(step),
                    format="%.4f" if step < 1 else "%.0f",
                    on_change=sync_slider, args=(feat,)
                )
                if st.button(f"Reset {feat.title()}", key=f"reset_{feat}"):
                    st.session_state.reset_trigger = feat
                    st.experimental_rerun()

                values[feat] = st.session_state[f"n_{feat}"]

with col_right:
    st.subheader("Feature-Level Malignancy Likelihood")
    records = []
    for feat, user_val in values.items():
        m = 0.05 * user_val
        slice_df = df[(df[feat] >= user_val - m) & (df[feat] <= user_val + m)]
        pct = 100 * slice_df["is_malignant"].mean() if not slice_df.empty else None
        records.append((feat, user_val, pct, len(slice_df)))

    lik_df = pd.DataFrame(records, columns=["Feature","Value","% Malignant","Count"]).dropna()
    if not lik_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lik_df["Feature"], y=lik_df["% Malignant"],
            mode="lines+markers+text",
            text=[f"{p:.1f}%" for p in lik_df["% Malignant"]],
            textposition="top center",
            line=dict(color="crimson", width=3)
        ))
        fig.update_layout(
            xaxis_title="Tumor Feature",
            yaxis_title="% of Similar Cases that were Malignant",
            yaxis_range=[0,100],
            height=500,
            margin=dict(l=10, r=10, t=10, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to show chart.")

    st.subheader("Diagnosis Estimate")
    ordered = [f for grp in FEATURE_GROUPS.values() for f in grp]
    X = np.array([[values[f] for f in ordered]])
    prob = pipe.predict_proba(X)[0,1]
    if prob >= 0.5:
        st.error(f"🚨 MALIGNANT: {prob:.1%}", icon="🚨")
    else:
        st.success(f"🫰 BENIGN: {(1-prob):.1%}", icon="✅")

    st.caption("For educational use only; not medical advice.")
