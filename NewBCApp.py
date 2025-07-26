import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path
from sklearn.datasets import load_breast_cancer

st.set_page_config(layout="wide")

# ──────────────────────────  Model & settings  ───────────────────────────────
MODEL_PATH = Path("breast_cancer_pipe_updated.pkl")
TEST_ACC   = 0.965

# Load breast cancer dataset for reference profiles
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Grouped input structure
FEATURE_GROUPS = {
    "Radius (mm)": [
        ("mean radius", "Mean radius", "Average nucleus distance to edge", 0.0, 50.0, 0.01, 14.127),
        ("worst radius", "Worst radius", "Maximum radius observed", 7.9, 36.0, 0.1, 16.27),
    ],
    "Perimeter (mm)": [
        ("mean perimeter", "Mean perimeter", "Average nucleus perimeter", 0.0, 300.0, 0.01, 91.969),
        ("worst perimeter", "Worst perimeter", "Maximum perimeter observed", 50.4, 251.2, 0.1, 107.3),
    ],
    "Area (mm²)": [
        ("mean area", "Mean area", "Average 2D tumor area", 0.0, 2500.0, 1.0, 654.889),
        ("worst area", "Worst area", "Maximum 2D tumor area", 185.2, 4254.0, 1.0, 880.6),
    ],
    "Concavity": [
        ("mean concavity", "Mean concavity", "Average inward curve depth", 0.0, 0.4268, 0.001, 0.0888),
        ("worst concavity", "Worst concavity", "Maximum inward curve depth", 0.0, 1.2520, 0.001, 0.2722),
    ],
    "Concave Points": [
        ("mean concave points", "Mean concave points", "Average edge indentations", 0.0, 0.2012, 0.001, 0.0489),
        ("worst concave points", "Worst concave points", "Maximum edge indentations", 0.0, 0.2910, 0.001, 0.1146),
    ],
    "Texture": [
        ("mean texture", "Mean texture", "Std-dev of gray values in nucleus", 0.0, 100.0, 0.01, 19.289),
    ]
}

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

pipe = load_model(MODEL_PATH)

# ─────────────────────────────  Page layout  ─────────────────────────────────
st.title("Breast Cancer ML Classifier 🩺")
st.caption(f"Model hold-out accuracy: {TEST_ACC:.1%}")

left_col, right_col = st.columns([2, 3], gap="large")

values = {}
with left_col:
    st.subheader("Adjust tumour characteristics")
    for group_title, feature_list in FEATURE_GROUPS.items():
        st.markdown(f"### {group_title}")
        if len(feature_list) == 1:
            # Single input
            cfg = feature_list[0]
            key, label, desc, vmin, vmax, step, avg = cfg
            st.markdown(f"<h4 style='margin-bottom:0.2rem'>{label}</h4>", unsafe_allow_html=True)
            st.caption(desc)
            avg_display = f"{avg:.4f}" if step < 1 else f"{avg:.0f}"
            st.caption(f"*Population average: {avg_display}*")
            s_col, n_col = st.columns([3, 1])
            slid_val = s_col.slider(label="", key=f"s_{key}", min_value=vmin, max_value=vmax,
                                    value=avg, step=step, label_visibility="collapsed")
            num_val = n_col.number_input(label="Exact", key=f"n_{key}", min_value=vmin, max_value=vmax,
                                         value=slid_val, step=step, format="%.4f" if step < 1 else "%.0f")
            values[key] = num_val
        else:
            left, right = st.columns(2, gap="large")
            for col, cfg in zip((left, right), feature_list):
                key, label, desc, vmin, vmax, step, avg = cfg
                with col:
                    st.markdown(f"<h4 style='margin-bottom:0.2rem'>{label}</h4>", unsafe_allow_html=True)
                    st.caption(desc)
                    avg_display = f"{avg:.4f}" if step < 1 else f"{avg:.0f}"
                    st.caption(f"*Population average: {avg_display}*")
                    s_col, n_col = st.columns([3, 1])
                    slid_val = s_col.slider(label="", key=f"s_{key}", min_value=vmin, max_value=vmax,
                                            value=avg, step=step, label_visibility="collapsed")
                    num_val = n_col.number_input(label="Exact", key=f"n_{key}", min_value=vmin, max_value=vmax,
                                                 value=slid_val, step=step, format="%.4f" if step < 1 else "%.0f")
                    values[key] = num_val

with right_col:
    st.subheader("Feature-Level Malignancy Likelihood")

    user_input_dict = {k: values[k] for k in values}
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
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Diagnosis Estimate")
    ordered_keys = [f[0] for group in FEATURE_GROUPS.values() for f in group]
    X = np.array([[values[k] for k in ordered_keys]])
    p = pipe.predict_proba(X)[0, 1]

    if p >= 0.5:
        st.error(
            f"🚨 **MALIGNANT**  \n"
            f"Probability: **{p:.1%}** "
            f"(≈ {p*100:.0f} out of 100 similar cases)",
            icon="🚨",
        )
    else:
        st.success(
            f"🩺 **BENIGN**  \n"
            f"Probability: **{1-p:.1%}** "
            f"(≈ {(1-p)*100:.0f} out of 100 similar cases)",
            icon="✅",
        )

    st.caption("Model is for educational use only and **does not replace professional medical advice.**")
