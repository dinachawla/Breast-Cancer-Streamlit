import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path
from sklearn.datasets import load_breast_cancer

st.set_page_config(layout="wide")

# Sticky behavior for the right column content
st.markdown("""
    <style>
    [data-testid="column"]:nth-of-type(3) > div {
        position: sticky;
        top: 80px;
        margin-bottom: auto;
        background-color: white;
        padding: 1rem 0.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        z-index: 2;
    }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("breast_cancer_pipe_11features.pkl")
TEST_ACC = 0.965

# Load dataset and model
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['is_malignant'] = (data.target == 0).astype(int)

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

pipe = load_model(MODEL_PATH)

# Pull coefficients out of the final classifier step (assumes a 'classifier' named step)
# so we can give per-feature directionality
try:
    model = pipe.named_steps['classifier']
    coefs = model.coef_[0]
except Exception:
    # fallback if your step is named differently
    model = pipe.steps[-1][1]
    coefs = getattr(model, 'coef_', [0]*len(df.columns))
    
SELECTED_FEATURES = [
    "mean radius", "worst radius",
    "mean perimeter", "worst perimeter",
    "mean area", "worst area",
    "mean concavity", "worst concavity",
    "mean concave points", "worst concave points",
    "mean texture"
]
feature_coefs = dict(zip(SELECTED_FEATURES, coefs))

# Compute percentile bounds
percentile_bounds = {}
for col in SELECTED_FEATURES:
    low = np.percentile(df[col], 5)
    high = np.percentile(df[col], 95)
    avg = df[col].mean()
    step = 0.001 if high - low < 10 else 0.1 if high - low < 100 else 1
    percentile_bounds[col] = (low, high, step, avg)

FEATURE_GROUPS = {
    "Radius (mm)": ["mean radius", "worst radius"],
    "Perimeter (mm)": ["mean perimeter", "worst perimeter"],
    "Area (mm²)": ["mean area", "worst area"],
    "Concavity": ["mean concavity", "worst concavity"],
    "Concave Points": ["mean concave points", "worst concave points"],
    "Texture": ["mean texture"]
}

# Session state for sliders + inputs
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = None

for key in SELECTED_FEATURES:
    if f"s_{key}" not in st.session_state:
        st.session_state[f"s_{key}"] = percentile_bounds[key][3]
        st.session_state[f"n_{key}"] = percentile_bounds[key][3]

if st.session_state.reset_trigger:
    rk = st.session_state.reset_trigger
    _, _, _, avg = percentile_bounds[rk]
    st.session_state[f"s_{rk}"] = avg
    st.session_state[f"n_{rk}"] = avg
    st.session_state.reset_trigger = None

def sync_slider(k): st.session_state[f"s_{k}"] = st.session_state[f"n_{k}"]
def sync_number(k): st.session_state[f"n_{k}"] = st.session_state[f"s_{k}"]

# UI
st.title("Breast Cancer ML Classifier 🩺")
st.caption(f"Model hold-out accuracy: {TEST_ACC:.1%}")
st.subheader("Adjust Tumor Characteristics")

left_col, right_col = st.columns([1,1], gap="large")

with left_col:
    values = {}
    for grp, keys in FEATURE_GROUPS.items():
        st.markdown(f"### {grp}")
        cols = st.columns(len(keys))
        for col, key in zip(cols, keys):
            with col:
                # description box
                coef = feature_coefs.get(key, 0)
                direction = "increases" if coef > 0 else "decreases"
                target = "malignancy risk" if coef > 0 else "benign likelihood"
                desc = f"As `{key}` {direction}, {target}."
                st.markdown(
                    f"<div style='background:#f5f5f5; padding:0.5rem; border-radius:0.5rem; font-size:0.9rem; margin-bottom:0.5rem'>{desc}</div>",
                    unsafe_allow_html=True
                )

                low, high, step, avg = percentile_bounds[key]
                st.markdown(f"<strong>{key.title()}</strong>", unsafe_allow_html=True)
                st.caption(f"*Population avg: {avg:.3f}*")

                st.slider(
                    label="", key=f"s_{key}",
                    min_value=float(low), max_value=float(high),
                    step=float(step), label_visibility="collapsed",
                    on_change=sync_number, args=(key,)
                )
                st.number_input(
                    "Exact", key=f"n_{key}",
                    min_value=float(low), max_value=float(high),
                    step=float(step),
                    format="%.4f" if step<1 else "%.0f",
                    on_change=sync_slider, args=(key,)
                )
                if st.button(f"Reset {key.title()}", key=f"reset_{key}"):
                    st.session_state.reset_trigger = key
                    st.experimental_rerun()

                values[key] = st.session_state[f"n_{key}"]

with right_col:
    st.subheader("Feature-Level Malignancy Likelihood")
    likelihoods = []
    for feature, user_val in values.items():
        m = 0.05 * user_val
        slice_df = df[(df[feature]>=user_val-m)&(df[feature]<=user_val+m)]
        pct = 100*slice_df["is_malignant"].mean() if not slice_df.empty else None
        likelihoods.append((feature, user_val, pct, len(slice_df)))

    lik_df = pd.DataFrame(likelihoods, columns=["Feature","User Value","% Malignant","Count"]).dropna()
    if not lik_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lik_df["Feature"], y=lik_df["% Malignant"],
            mode="lines+markers+text",
            text=[f"{p:.1f}%" for p in lik_df["% Malignant"]],
            textposition="top center", line=dict(width=3)
        ))
        fig.update_layout(yaxis_range=[0,100], margin=dict(b=40))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for chart.")

    st.subheader("Diagnosis Estimate")
    order = [k for keys in FEATURE_GROUPS.values() for k in keys]
    X = np.array([[values[k] for k in order]])
    p = pipe.predict_proba(X)[0,1]

    if p>=0.5:
        st.error(f"MALIGNANT: {p:.1%} (≈{p*100:.0f}%)", icon="🚨")
    else:
        st.success(f"BENIGN: {(1-p):.1%} (≈{(1-p)*100:.0f}%)", icon="✅")

    st.caption("For educational use only; not medical advice.")
