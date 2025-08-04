import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path
from sklearn.datasets import load_breast_cancer

# --------------------------------------
# CONFIG & SETUP
# --------------------------------------
st.set_page_config(layout="wide")

# Sticky CSS for right column
st.markdown(
    """
    <style>
    .sticky-right {
        position: -webkit-sticky;
        position: sticky;
        top: 1rem;
        align-self: start;
        background-color: white;
        padding-top: 0.5rem;
        z-index: 10;
    }
    .metric-desc {
        font-size: 0.85rem;
        color: gray;
    }
    </style>
    """,
    unsafe_allow_html=True
)

MODEL_PATH = Path("breast_cancer_pipe_11features.pkl")
TEST_ACC = 0.965

# Load data & model
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
# In the sklearn dataset: 0=malignant, 1=benign. Flip so 1=malignant
df['target'] = 1 - data.target
pipe = joblib.load(MODEL_PATH)

# Extract coefficients for descriptions
lr = pipe.named_steps['logisticregression']
coefs = dict(zip(pipe.feature_names_in_, lr.coef_[0]))

# Build tooltip/descriptions
BASE_TOOLTIPS = {
    "mean radius": "Average distance from center to tumor edge",
    "mean perimeter": "Average boundary length of the tumor",
    "mean area": "Average surface area of the tumor",
    "worst radius": "Largest observed distance from center to edge",
    "worst perimeter": "Longest boundary length observed",
    "worst area": "Largest surface area measured",
    "mean concavity": "Average severity of concave portions",
    "worst concavity": "Maximum concavity observed",
    "mean concave points": "Average number of concave points on contour",
    "worst concave points": "Max number of concave points",
    "mean texture": "Standard deviation of gray-scale values"
}
TOOLTIPS = {}
for feat, desc in BASE_TOOLTIPS.items():
    sign = 'increases' if coefs[feat] > 0 else 'decreases'
    TOOLTIPS[feat] = f"{desc}. Higher values {sign} malignancy risk."

SELECTED_FEATURES = list(TOOLTIPS.keys())

# Compute slider bounds
def compute_bounds(col):
    low = np.percentile(df[col], 5)
    high = np.percentile(df[col], 95)
    avg = df[col].mean()
    rng = high - low
    step = 0.001 if rng < 10 else 0.1 if rng < 100 else 1
    return low, high, step, avg

percentile_bounds = {col: compute_bounds(col) for col in SELECTED_FEATURES}

# Group features: mean vs worst
MEAN_FEATURES = [f for f in SELECTED_FEATURES if f.startswith('mean')]
WORST_FEATURES = [f for f in SELECTED_FEATURES if f.startswith('worst')]

# --------------------------------------
# SESSION STATE
# --------------------------------------
if 'vals' not in st.session_state:
    st.session_state.vals = {f: percentile_bounds[f][3] for f in SELECTED_FEATURES}

# --------------------------------------
# UI
# --------------------------------------
st.title("CURA - Breast Cancer ML Classifier 🩺")
st.caption(f"Model hold-out accuracy: {TEST_ACC:.1%}")

left_col, right_col = st.columns([1, 1], gap='large')

# --- LEFT: Manual Sliders ---
with left_col:
    st.subheader("Adjust Tumor Characteristics")
    
    # Mean feature group
    st.markdown("### Mean-Based Metrics")
    for i in range(0, len(MEAN_FEATURES), 2):
        row_feats = MEAN_FEATURES[i:i+2]
        cols = st.columns(len(row_feats))
        for col_area, feat in zip(cols, row_feats):
            low, high, step, avg = percentile_bounds[feat]
            with col_area:
                st.markdown(f"**{feat.title()}**")
                st.caption(f"_Population avg: {avg:.3f}_")
                val = st.slider(
                    label="", key=f"s_{feat}",
                    min_value=float(low), max_value=float(high),
                    step=float(step), value=st.session_state.vals[feat]
                )
                st.session_state.vals[feat] = val
                st.markdown(f"<div class='metric-desc'>{TOOLTIPS[feat]}</div>", unsafe_allow_html=True)

    # Worst feature group
    st.markdown("### Worst-Based Metrics")
    for i in range(0, len(WORST_FEATURES), 2):
        row_feats = WORST_FEATURES[i:i+2]
        cols = st.columns(len(row_feats))
        for col_area, feat in zip(cols, row_feats):
            low, high, step, avg = percentile_bounds[feat]
            with col_area:
                st.markdown(f"**{feat.title()}**")
                st.caption(f"_Population avg: {avg:.3f}_")
                val = st.slider(
                    label="", key=f"s_{feat}",
                    min_value=float(low), max_value=float(high),
                    step=float(step), value=st.session_state.vals[feat]
                )
                st.session_state.vals[feat] = val
                st.markdown(f"<div class='metric-desc'>{TOOLTIPS[feat]}</div>", unsafe_allow_html=True)

# --- RIGHT: Sticky Chart & Prediction ---
with right_col:
    st.markdown("<div class='sticky-right'>", unsafe_allow_html=True)
    
    # Feature-level model-based likelihood
    st.subheader("Feature-Level Malignancy Likelihood (Model)")
    feats, vals, probs = [], [], []
    default_means = {f: df[f].mean() for f in pipe.feature_names_in_}
    for feat, user_val in st.session_state.vals.items():
        inp = default_means.copy()
        inp[feat] = user_val
        Xf = pd.DataFrame([inp])[pipe.feature_names_in_]
        p_feat = pipe.predict_proba(Xf)[0,1]
        feats.append(feat)
        vals.append(user_val)
        probs.append(100 * p_feat)

    chart_df = pd.DataFrame({"Feature": feats, "% Malignant": probs})
    fig = go.Figure(go.Bar(
        x=chart_df['Feature'], y=chart_df['% Malignant']
    ))
    fig.update_layout(
        yaxis=dict(range=[0,100]), height=400,
        margin=dict(l=10,r=10,t=10,b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Global prediction
    st.subheader("Diagnosis Estimate")
    Xall = pd.DataFrame([st.session_state.vals])[pipe.feature_names_in_]
    p_all = pipe.predict_proba(Xall)[0,1]
    if p_all >= 0.5:
        st.error(f"🚨 **MALIGNANT** {p_all:.1%}", icon="🚨")
    else:
        st.success(f"✅ **BENIGN** {(1-p_all):.1%}", icon="✅")

    st.caption("Model is for educational use only and does not replace professional medical advice.")
    st.markdown("</div>", unsafe_allow_html=True)
