import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer

# ──────────────────────────  Model & settings  ───────────────────────────────
MODEL_PATH = Path("breast_cancer_pipe_updated.pkl")
TEST_ACC   = 0.965   # hold-out accuracy

# Load breast cancer dataset for reference profiles
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Grouped input structure for paired comparison layout
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

# ─────────────────────────────  Page header  ─────────────────────────────────
st.title("Breast Cancer ML Classifier 🩺")
st.markdown(
    "Estimate whether a breast-tumour sample is **benign** or **malignant**. "
    "Move each slider *or* type an exact value."
)
st.caption(f"Model hold-out accuracy: {TEST_ACC:.1%}")
st.subheader("Adjust tumour characteristics")

# ────────────────────────  Input pairs by feature group  ─────────────────────
values = {}
for group_title, feature_list in FEATURE_GROUPS.items():
    st.markdown(f"### {group_title}")
    if len(feature_list) == 1:
        feature_list = [feature_list[0], None]

    left, right = st.columns(2, gap="large")
    for col, cfg in zip((left, right), feature_list):
        if cfg is None:
            continue
        key, label, desc, vmin, vmax, step, avg = cfg
        with col:
            st.markdown(f"<h4 style='margin-bottom:0.2rem'>{label}</h4>", unsafe_allow_html=True)
            st.caption(desc)
            avg_display = f"{avg:.4f}" if step < 1 else f"{avg:.0f}"
            st.caption(f"*Population average: {avg_display}*")

            s_col, n_col = st.columns([3, 1])
            with s_col:
                slid_val = st.slider(
                    label="", key=f"s_{key}",
                    min_value=vmin, max_value=vmax,
                    value=avg, step=step,
                    label_visibility="collapsed"
                )
            with n_col:
                num_val = st.number_input(
                    label="Exact", key=f"n_{key}",
                    min_value=vmin, max_value=vmax,
                    value=slid_val, step=step,
                    format="%.4f" if step < 1 else "%.0f"
                )
            values[key] = num_val

# ───────────────  Live feature comparison chart with toggle ───────────────────────
st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
chart_type = st.radio("Choose chart type:", ["Bar Chart", "Line Graph"], horizontal=True)

ordered_keys = [f[0] for group in FEATURE_GROUPS.values() for f in group]
user_input = np.array([values[k] for k in ordered_keys])

benign_profile = df[df['target'] == 1][ordered_keys].mean()
malignant_profile = df[df['target'] == 0][ordered_keys].mean()

scaler = MinMaxScaler()
scaled = scaler.fit_transform([benign_profile, malignant_profile, user_input])
benign_scaled, malignant_scaled, user_scaled = scaled

profile_df = pd.DataFrame({
    'Feature': ordered_keys * 3,
    'Value': np.concatenate([benign_scaled, malignant_scaled, user_scaled]),
    'Profile': ['Benign'] * len(ordered_keys) +
               ['Malignant'] * len(ordered_keys) +
               ['Your Input'] * len(ordered_keys)
})

if chart_type == "Bar Chart":
    fig = px.bar(
        profile_df,
        x='Feature',
        y='Value',
        color='Profile',
        barmode='group',
        title='Live Tumor Feature Comparison (Normalized)',
        height=500
    )
    fig.update_layout(xaxis_tickangle=-45)
else:
    fig = go.Figure()
    for profile in ['Benign', 'Malignant', 'Your Input']:
        profile_data = profile_df[profile_df['Profile'] == profile]
        fig.add_trace(go.Scatter(
            x=profile_data['Feature'],
            y=profile_data['Value'],
            mode='lines+markers',
            name=profile
        ))
    fig.update_layout(
        title='Tumor Profile Comparison (Line Chart)',
        xaxis_title='Feature',
        yaxis_title='Normalized Value',
        height=500
    )

st.plotly_chart(fig, use_container_width=True)

# ───────────────  Prediction & result card  ───────────────────────
st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

if st.button("Classify"):
    X = np.array([[values[k] for k in ordered_keys]])
    p = pipe.predict_proba(X)[0, 1]  # probability malignant

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
