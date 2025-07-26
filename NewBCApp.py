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

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
percentiles = df.describe(percentiles=[.05, .95]).T

FEATURE_GROUPS = {
    "Radius (mm)": [
        ("mean radius", "Mean radius", "Average nucleus distance to edge", 0.01, 14.127),
        ("worst radius", "Worst radius", "Maximum radius observed", 0.1, 16.27),
    ],
    "Perimeter (mm)": [
        ("mean perimeter", "Mean perimeter", "Average nucleus perimeter", 0.01, 91.969),
        ("worst perimeter", "Worst perimeter", "Maximum perimeter observed", 0.1, 107.3),
    ],
    "Area (mm²)": [
        ("mean area", "Mean area", "Average 2D tumor area", 1.0, 654.889),
        ("worst area", "Worst area", "Maximum 2D tumor area", 1.0, 880.6),
    ],
    "Concavity": [
        ("mean concavity", "Mean concavity", "Average inward curve depth", 0.001, 0.0888),
        ("worst concavity", "Worst concavity", "Maximum inward curve depth", 0.001, 0.2722),
    ],
    "Concave Points": [
        ("mean concave points", "Mean concave points", "Average edge indentations", 0.001, 0.0489),
        ("worst concave points", "Worst concave points", "Maximum edge indentations", 0.001, 0.1146),
    ],
    "Texture": [
        ("mean texture", "Mean texture", "Std-dev of gray values in nucleus", 0.01, 19.289),
    ]
}

def get_bounds(feature_name):
    lower = percentiles.loc[feature_name]["5%"]
    upper = percentiles.loc[feature_name]["95%"]
    return lower, upper

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

pipe = load_model(MODEL_PATH)

st.title("Breast Cancer ML Classifier 🩺")
st.caption(f"Model hold-out accuracy: {TEST_ACC:.1%}")
st.subheader("Adjust Tumor Characteristics")

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    values = {}
    for group_title, feature_list in FEATURE_GROUPS.items():
        st.markdown(f"### {group_title}")
        if len(feature_list) == 2:
            f1, f2 = feature_list
            col1, col2 = st.columns(2, gap="medium")
            for col, cfg in zip((col1, col2), (f1, f2)):
                key, label, desc, step, avg = cfg
                vmin, vmax = get_bounds(key)
                with col:
                    st.markdown(f"<h4 style='margin-bottom:0.2rem'>{label}</h4>", unsafe_allow_html=True)
                    st.caption(desc)
                    avg_display = f"{avg:.4f}" if step < 1 else f"{avg:.0f}"
                    st.caption(f"*Population average: {avg_display}*")
                    slider_val = st.slider(label="", key=f"s_{key}", min_value=vmin, max_value=vmax, value=avg, step=step, label_visibility="collapsed")
                    safe_val = min(max(slider_val, vmin), vmax)
                    num_val = st.number_input(label="Exact", key=f"n_{key}", min_value=vmin, max_value=vmax, value=safe_val, step=step, format="%.4f" if step < 1 else "%.0f")
                    if st.button(f"Reset {label}", key=f"reset_{key}"):
                        st.session_state[f"s_{key}"] = avg
                        st.session_state[f"n_{key}"] = avg
                    values[key] = num_val
        else:
            for cfg in feature_list:
                key, label, desc, step, avg = cfg
                vmin, vmax = get_bounds(key)
                st.markdown(f"<h4 style='margin-bottom:0.2rem'>{label}</h4>", unsafe_allow_html=True)
                st.caption(desc)
                avg_display = f"{avg:.4f}" if step < 1 else f"{avg:.0f}"
                st.caption(f"*Population average: {avg_display}*")
                slider_val = st.slider(label="", key=f"s_{key}", min_value=vmin, max_value=vmax, value=avg, step=step, label_visibility="collapsed")
                safe_val = min(max(slider_val, vmin), vmax)
                num_val = st.number_input(label="Exact", key=f"n_{key}", min_value=vmin, max_value=vmax, value=safe_val, step=step, format="%.4f" if step < 1 else "%.0f")
                if st.button(f"Reset {label}", key=f"reset_{key}"):
                    st.session_state[f"s_{key}"] = avg
                    st.session_state[f"n_{key}"] = avg
                values[key] = num_val

with right_col:
    st.subheader("Feature-Level Malignancy Likelihood")

    user_input_dict = {k: v for k, v in values.items()}
    likelihoods = []
    for feature, user_val in user_input_dict.items():
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

    st.subheader("Diagnosis Estimate")
    ordered_keys = [f[0] for group in FEATURE_GROUPS.values() for f in group]
    X = np.array([[values[k] for k in ordered_keys]])
    p = pipe.predict_proba(X)[0, 1]
    if p >= 0.5:
        st.error(f"🚨 **MALIGNANT**  \nProbability: **{p:.1%}** (≈ {p*100:.0f} out of 100 similar cases)", icon="🚨")
    else:
        st.success(f"🩺 **BENIGN**  \nProbability: **{1-p:.1%}** (≈ {(1-p)*100:.0f} out of 100 similar cases)", icon="✅")
    st.caption("Model is for educational use only and **does not replace professional medical advice.**")
