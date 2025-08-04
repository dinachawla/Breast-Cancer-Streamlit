import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path
from sklearn.datasets import load_breast_cancer

st.set_page_config(layout="wide")

# Sticky CSS and metric description style
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
    .custom-box { background-color: #1e1e1e; padding: 1rem; border-radius: 0.5rem; }
    .custom-title { color: white; margin-bottom: 0.2rem; }
    .custom-caption { color: lightgray; font-size: 0.85rem; }
    .metric-desc { color: lightgray; font-size: 0.8rem; margin-top: 0.25rem; margin-bottom: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("breast_cancer_pipe_11features.pkl")
TEST_ACC = 0.965

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
# scikit: 0=malignant, 1=benign
pipe = joblib.load(MODEL_PATH)

# Dynamic tooltips based on model coefficients
lr = pipe.named_steps['logisticregression']
coefs = dict(zip(pipe.feature_names_in_, lr.coef_[0]))
BASE_TOOLTIPS = {
    "mean radius": "Average distance from center to tumor edge",
    "worst radius": "Largest observed distance from center to edge",
    "mean perimeter": "Average boundary length of the tumor",
    "worst perimeter": "Longest boundary length observed",
    "mean area": "Average surface area of the tumor",
    "worst area": "Largest surface area measured",
    "mean concavity": "Average severity of concave portions",
    "worst concavity": "Maximum concavity observed",
    "mean concave points": "Average number of concave points on contour",
    "worst concave points": "Max number of concave points",
    "mean texture": "Standard deviation of gray-scale values"
}
TOOLTIPS = {
    feat: f"{desc}. Higher values {'increase' if coefs.get(feat,0)>0 else 'decrease'} malignancy risk."
    for feat, desc in BASE_TOOLTIPS.items()
}
SELECTED_FEATURES = list(TOOLTIPS.keys())

# Compute percentile bounds for sliders
percentile_bounds = {}
for col in SELECTED_FEATURES:
    low = np.percentile(df[col], 5)
    high = np.percentile(df[col], 95)
    avg = df[col].mean()
    rng = high - low
    step = 0.001 if rng < 10 else 0.1 if rng < 100 else 1
    percentile_bounds[col] = (low, high, step, avg)

# Group into mean and worst
FEATURE_GROUPS = {
    "Mean-Based Metrics":  [f for f in SELECTED_FEATURES if f.startswith('mean')],
    "Worst-Based Metrics": [f for f in SELECTED_FEATURES if f.startswith('worst')]
}

# Session-state initialization
if 'reset_trigger' not in st.session_state:
    st.session_state.reset_trigger = None
for key in SELECTED_FEATURES:
    if f"s_{key}" not in st.session_state or f"n_{key}" not in st.session_state:
        _, _, _, avg = percentile_bounds[key]
        st.session_state[f"s_{key}"] = avg
        st.session_state[f"n_{key}"] = avg
if st.session_state.reset_trigger:
    reset_key = st.session_state.reset_trigger
    _, _, _, avg = percentile_bounds[reset_key]
    st.session_state[f"s_{reset_key}"] = avg
    st.session_state[f"n_{reset_key}"] = avg
    st.session_state.reset_trigger = None

def sync_slider(key):
    st.session_state[f"s_{key}"] = st.session_state[f"n_{key}"]
def sync_number_input(key):
    st.session_state[f"n_{key}"] = st.session_state[f"s_{key}"]

st.title("CURA - Breast Cancer ML Classifier 🩺")
st.caption(f"Model hold-out accuracy: {TEST_ACC:.1%}")
st.subheader("Adjust Tumor Characteristics")

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    values = {}
    for group_title, keys in FEATURE_GROUPS.items():
        st.markdown(f"### {group_title}")
        for i in range(0, len(keys), 2):
            row_feats = keys[i:i+2]
            cols = st.columns(len(row_feats))
            for col_feat, feat in zip(cols, row_feats):
                with col_feat:
                    low, high, step, avg = percentile_bounds[feat]
                    st.markdown("<div class='custom-box'>", unsafe_allow_html=True)
                    st.markdown(f"<h4 class='custom-title'>{feat.title()}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p class='custom-caption'>Avg: {avg:.3f}</p>", unsafe_allow_html=True)
                    slider_val = st.slider(
                        label="", key=f"s_{feat}",
                        min_value=float(low), max_value=float(high),
                        step=float(step), value=st.session_state[f"s_{feat}"],
                        on_change=sync_number_input, args=(feat,)
                    )
                    number_val = st.number_input(
                        label="Exact", key=f"n_{feat}",
                        min_value=float(low), max_value=float(high),
                        step=float(step), value=slider_val,
                        on_change=sync_slider, args=(feat,)
                    )
                    st.markdown(f"<div class='metric-desc'>{TOOLTIPS[feat]}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    values[feat] = number_val

with right_col:
    st.subheader("Feature-Level Malignancy Likelihood")
    feats, probs = [], []
    base_vals = {f: df[f].mean() for f in SELECTED_FEATURES}
    for feat, user_val in values.items():
        inp = base_vals.copy(); inp[feat] = user_val
        Xf = pd.DataFrame([inp])[pipe.feature_names_in_]
        p_feat = pipe.predict_proba(Xf)[0, 1] * 100
        feats.append(feat)
        probs.append(p_feat)
    likelihood_df = pd.DataFrame({"Feature": feats, "% Malignant": probs})
    fig = go.Figure(go.Scatter(
        x=likelihood_df['Feature'], y=likelihood_df['% Malignant'],
        mode='lines+markers+text', text=[f"{p:.1f}%" for p in likelihood_df['% Malignant']],
        textposition='top center', line=dict(color='crimson', width=3)
    ))
    fig.update_layout(
        xaxis_title='Tumor Feature', yaxis_title='% of Similar Cases that were Malignant',
        yaxis_range=[0, 100], height=500, margin=dict(l=10, r=10, t=10, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Diagnosis Estimate")
    input_df = pd.DataFrame([values])[pipe.feature_names_in_]
    p = pipe.predict_proba(input_df)[0, 1]
    if p >= 0.85:
        confidence = "High Confidence"
    elif p >= 0.6:
        confidence = "Moderate Confidence"
    else:
        confidence = "Low Confidence"
    if p >= 0.5:
        st.error(f"**MALIGNANT**  \nProbability: **{p:.1%}** ({confidence})", icon="🚨")
    else:
        st.success(f"**BENIGN**  \nProbability: **{(1-p):.1%}** ({confidence})", icon="✅")

    diffs = {k: abs(values[k] - df[k].mean()) for k in values}
    top_feature = max(diffs, key=diffs.get)
    st.markdown(f"**Most Influential Feature:** {top_feature.title()} — largest deviation from average")
    st.caption("Model is for educational use only and **does not replace professional medical advice.**")
