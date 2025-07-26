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

# Load dataset and model
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

pipe = load_model(MODEL_PATH)

# Compute percentiles
percentile_bounds = {}
for col in data.feature_names:
    low = np.percentile(df[col], 5)
    high = np.percentile(df[col], 95)
    avg = df[col].mean()
    step = 0.001 if high - low < 10 else 0.1 if high - low < 100 else 1
    percentile_bounds[col] = (low, high, step, avg)

# Define grouped features
FEATURE_GROUPS = {
    "Radius (mm)": ["mean radius", "worst radius"],
    "Perimeter (mm)": ["mean perimeter", "worst perimeter"],
    "Area (mm²)": ["mean area", "worst area"],
    "Concavity": ["mean concavity", "worst concavity"],
    "Concave Points": ["mean concave points", "worst concave points"],
    "Texture": ["mean texture"]
}

# Init session state
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = None

for key in data.feature_names:
    if f"s_{key}" not in st.session_state or f"n_{key}" not in st.session_state:
        _, _, _, avg = percentile_bounds[key]
        st.session_state[f"s_{key}"] = avg
        st.session_state[f"n_{key}"] = avg

# Handle reset (safely)
if st.session_state.reset_trigger is not None:
    reset_key = st.session_state.reset_trigger
    if reset_key in percentile_bounds:
        _, _, _, avg = percentile_bounds[reset_key]
        st.session_state[f"s_{reset_key}"] = avg
        st.session_state[f"n_{reset_key}"] = avg
    st.session_state.reset_trigger = None

# Sync logic
def sync_slider(key):
    st.session_state[f"s_{key}"] = st.session_state[f"n_{key}"]

def sync_number_input(key):
    st.session_state[f"n_{key}"] = st.session_state[f"s_{key}"]

# UI
st.title("Breast Cancer ML Classifier 🩺")
st.caption(f"Model hold-out accuracy: {TEST_ACC:.1%}")
st.subheader("Adjust Tumor Characteristics")

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    with st.container():
        st.markdown(
            """
            <style>
            .scrollbox {
                max-height: 600px;
                overflow-y: auto;
                padding-right: 1rem;
            }
            </style>
            <div class="scrollbox">
            """,
            unsafe_allow_html=True
        )

        values = {}
        for group_title, keys in FEATURE_GROUPS.items():
            st.markdown(f"### {group_title}")
            cols = st.columns(len(keys))  # Always safe
            for col, key in zip(cols, keys):
                with col:
                    low, high, step, avg = percentile_bounds[key]
                    st.markdown(f"<h4 style='margin-bottom:0.2rem'>{key.title()}</h4>", unsafe_allow_html=True)
                    st.caption(f"*Population average: {avg:.3f}*")

                    st.slider(
                        label="", key=f"s_{key}",
                        min_value=float(low), max_value=float(high),
                        step=float(step), label_visibility="collapsed",
                        on_change=sync_number_input, args=(key,)
                    )

                    st.number_input(
                        label="Exact", key=f"n_{key}",
                        min_value=float(low), max_value=float(high),
                        step=float(step), format="%.4f" if step < 1 else "%.0f",
                        on_change=sync_slider, args=(key,)
                    )

                    if st.button(f"Reset {key.title()}", key=f"reset_{key}"):
                        st.session_state.reset_trigger = key
                        st.experimental_rerun()

                    values[key] = st.session_state[f"n_{key}"]

        st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.subheader("Feature-Level Malignancy Likelihood")
    likelihoods = []
    for feature, user_val in values.items():
        margin = 0.05 * user_val
        min_val = user_val - margin
        max_val = user_val + margin
        nearby_cases = df[(df[feature] >= min_val) & (df[feature] <= max_val)]
        if not nearby_cases.empty:
            malignant_pct = 100 * (1 - nearby_cases['target'].mean())
        else:
            malignant_pct = None
        likelihoods.append((feature, user_val, malignant_pct, len(nearby_cases)))

    likelihood_df = pd.DataFrame(likelihoods, columns=["Feature", "User Value", "% Malignant", "Cases in Range"])
    filtered_df = likelihood_df.dropna()

    if not filtered_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df['Feature'],
            y=filtered_df['% Malignant'],
            mode='lines+markers+text',
            name='% Malignant',
            line=dict(color='crimson', width=3),
            text=[f"{p:.1f}%" for p in filtered_df['% Malignant']],
            textposition="top center"
        ))
        fig.update_layout(
            title='Estimated Malignancy Likelihood per Feature',
            xaxis_title='Tumor Feature',
            yaxis_title='% of Similar Cases that were Malignant',
            yaxis_range=[0, 100],
            height=500,
            margin=dict(l=10, r=10, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to show malignancy likelihood chart.")

    st.subheader("Diagnosis Estimate")
    ordered_keys = [k for keys in FEATURE_GROUPS.values() for k in keys]
    X = np.array([[values[k] for k in ordered_keys]])
    p = pipe.predict_proba(X)[0, 1]
    if p >= 0.5:
        st.error(f"🚨 **MALIGNANT**  \nProbability: **{p:.1%}** (≈ {p*100:.0f} out of 100 similar cases)", icon="🚨")
    else:
        st.success(f"🩺 **BENIGN**  \nProbability: **{1-p:.1%}** (≈ {(1-p)*100:.0f} out of 100 similar cases)", icon="✅")
    st.caption("Model is for educational use only and **does not replace professional medical advice.**")
