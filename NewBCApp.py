import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path
from sklearn.datasets import load_breast_cancer

st.set_page_config(layout="wide")

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

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = 1 - data.target

SELECTED_FEATURES = [
    "mean radius", "worst radius",
    "mean perimeter", "worst perimeter",
    "mean area", "worst area",
    "mean concavity", "worst concavity",
    "mean concave points", "worst concave points",
    "mean texture"
]

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

pipe = load_model(MODEL_PATH)

percentile_bounds = {}
for col in SELECTED_FEATURES:
    low = np.percentile(df[col], 5)
    high = np.percentile(df[col], 95)
    avg = df[col].mean()
    step = 0.001 if high - low < 10 else 0.1 if high - low < 100 else 1
    percentile_bounds[col] = (low, high, step, avg)

FEATURE_GROUPS = {
    "Core Predictive Features": ["mean radius", "mean perimeter", "mean area"],
    "Radius (mm)": ["worst radius"],
    "Perimeter (mm)": ["worst perimeter"],
    "Area (mm²)": ["worst area"],
    "Concavity": ["mean concavity", "worst concavity"],
    "Concave Points": ["mean concave points", "worst concave points"],
    "Texture": ["mean texture"]
}

TOOLTIPS = {
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

if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = None

for key in SELECTED_FEATURES:
    if f"s_{key}" not in st.session_state or f"n_{key}" not in st.session_state:
        _, _, _, avg = percentile_bounds[key]
        st.session_state[f"s_{key}"] = avg
        st.session_state[f"n_{key}"] = avg

if st.session_state.reset_trigger is not None:
    reset_key = st.session_state.reset_trigger
    if reset_key in percentile_bounds:
        _, _, _, avg = percentile_bounds[reset_key]
        st.session_state[f"s_{reset_key}"] = avg
        st.session_state[f"n_{reset_key}"] = avg
    st.session_state.reset_trigger = None

def sync_slider(key):
    st.session_state[f"s_{key}"] = st.session_state[f"n_{key}"]

def sync_number_input(key):
    st.session_state[f"n_{key}"] = st.session_state[f"s_{key}"]

st.title("Breast Cancer ML Classifier 🩺")
st.caption(f"Model hold-out accuracy: {TEST_ACC:.1%}")
st.subheader("Adjust Tumor Characteristics")

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    values = {}
    for group_title, keys in FEATURE_GROUPS.items():
        st.markdown(f"### {group_title}")
        cols = st.columns(len(keys))
        for col, key in zip(cols, keys):
            with col:
                low, high, step, avg = percentile_bounds[key]
                st.markdown(
                    f"""
                    <div style='background-color:#1e1e1e; padding:1rem; border-radius:0.5rem;'>
                    <h4 style='color:white; margin-bottom:0.2rem'>{key.title()}</h4>
                    <p style='color:lightgray; font-size:0.85rem;'>{TOOLTIPS.get(key, '')} (Avg: {avg:.3f})</p>
                    """,
                    unsafe_allow_html=True
                )

                slider_val = st.slider(
                    label="", key=f"s_{key}",
                    min_value=float(low), max_value=float(high),
                    step=float(step), label_visibility="collapsed",
                    on_change=sync_number_input, args=(key,)
                )

                number_val = st.number_input(
                    label="Exact", key=f"n_{key}",
                    min_value=float(low), max_value=float(high),
                    step=float(step), format="%.4f" if step < 1 else "%.0f",
                    on_change=sync_slider, args=(key,)
                )

                values[key] = number_val

                if st.button(f"Reset {key.title()}", key=f"reset_{key}"):
                    st.session_state.reset_trigger = key
                    st.experimental_rerun()

                st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.subheader("Feature-Level Malignancy Likelihood")
    likelihoods = []
    for feature, user_val in values.items():
        margin = 0.05 * user_val
        min_val = user_val - margin
        max_val = user_val + margin
        nearby_cases = df[(df[feature] >= min_val) & (df[feature] <= max_val)]
        malignant_pct = 100 * nearby_cases['target'].mean() if not nearby_cases.empty else None
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
            xaxis_title='Tumor Feature',
            yaxis_title='% of Similar Cases that were Malignant',
            yaxis_range=[0, 100],
            height=500,
            margin=dict(l=10, r=10, t=10, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to show malignancy likelihood chart.")

    st.subheader("Diagnosis Estimate")
    ordered_keys = [k for keys in FEATURE_GROUPS.values() for k in keys]
    X = pd.DataFrame([[values[k] for k in ordered_keys]], columns=ordered_keys)
    p = pipe.predict_proba(X)[0, 1]

    if p >= 0.85:
        confidence = "High Confidence"
    elif p >= 0.6:
        confidence = "Moderate Confidence"
    else:
        confidence = "Low Confidence"

    if p >= 0.5:
        st.error(f"🚨 **MALIGNANT**  \nProbability: **{p:.1%}** ({confidence})", icon="🚨")
    else:
        st.success(f"✅ **BENIGN**  \nProbability: **{1 - p:.1%}** ({confidence})", icon="✅")

    diffs = {k: abs(values[k] - df[k].mean()) for k in values}
    top_feature = max(diffs, key=diffs.get)
    st.markdown(f"**Most Influential Feature:** {top_feature.title()} — largest deviation from average")
    st.caption("Model is for educational use only and **does not replace professional medical advice.**")
