import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path
from sklearn.datasets import load_breast_cancer

st.set_page_config(layout="wide")

# Sticky styling + theme-adaptive visible borders for metric boxes
st.markdown("""
    <style>
    [data-testid="column"]:nth-of-type(3) > div {
        position: sticky;
        top: 80px;
        background-color: var(--background-color);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        z-index: 2;
    }
    .metric-box {
        background-color: transparent;
        border: 1.5px solid rgba(0, 0, 0, 0.15); /* Light mode */
        color: var(--text-color);
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    /* Dark mode detection */
    @media (prefers-color-scheme: dark) {
        .metric-box {
            border: 1.5px solid rgba(255, 255, 255, 0.2); /* Dark mode */
        }
    }
    .metric-box h4 {
        margin: 0 0 0.25rem 0;
        font-size: 1rem;
        font-weight: bold;
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

# Pull out final-estimator coefficients
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

# Friendly explanations for each metric
METRIC_DESCRIPTIONS = {
    "mean radius":       "Average distance from tumor center to its boundary.",
    "worst radius":      "Largest radius observed, highlighting outlier growth.",
    "mean perimeter":    "Average length of the tumor boundary.",
    "worst perimeter":   "Maximum boundary length, indicating irregular expansion.",
    "mean area":         "Average surface area of tumor cells.",
    "worst area":        "Largest area observed, showing aggressive cell clusters.",
    "mean concavity":    "Average concavity of the tumor outline; higher = more indentations.",
    "worst concavity":   "Deepest indentations on the tumor boundary.",
    "mean concave points":"Average count of concave points on the tumor edge.",
    "worst concave points":"Maximum concave-point count, marking uneven growth.",
    "mean texture":      "Average variation in gray-scale values, reflecting cell uniformity."
}

# Precompute percentile bounds
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

# Session state init
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
st.title("Cura – Breast Cancer ML Classifier 🩺")
st.caption(f"Model hold-out accuracy: {TEST_ACC:.1%}")
st.subheader("Adjust Tumor Characteristics")

left_col, right_col = st.columns([1,1], gap="large")

with left_col:
    values = {}
    for group_title, feats in FEATURE_GROUPS.items():
        st.markdown(f"### {group_title}")
        cols = st.columns(len(feats))
        for col, feat in zip(cols, feats):
            with col:
                coef = feature_coefs[feat]
                direction = "Increasing" if coef > 0 else "Decreasing"
                low, high, step, avg = percentile_bounds[feat]
                desc = METRIC_DESCRIPTIONS[feat]

                st.markdown(f"""
<div class='metric-box'>
  <h4>{feat.title()}</h4>
  <p>{desc}</p>
  <p><em>Population average: {avg:.3f}</em></p>
  <p><strong>{direction}</strong> {feat} indicates malignancy.</p>
</div>
""", unsafe_allow_html=True)

                st.slider(
                    "", key=f"s_{feat}",
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

with right_col:
    st.subheader("Feature-Level Malignancy Likelihood")
    likelihoods = []
    for feat, user_val in values.items():
        m = 0.05 * user_val
        nearby = df[(df[feat] >= user_val - m) & (df[feat] <= user_val + m)]
        pct = 100 * nearby["is_malignant"].mean() if not nearby.empty else None
        if pct is not None and feature_coefs[feat] < 0:
            pct = 100 - pct
        likelihoods.append((feat, user_val, pct, len(nearby)))

    lik_df = pd.DataFrame(
        likelihoods,
        columns=["Feature","User Value","% Malignancy Indicator","Cases"]
    ).dropna()

    if not lik_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lik_df["Feature"],
            y=lik_df["% Malignancy Indicator"],
            mode="lines+markers+text",
            text=[f"{p:.1f}%" for p in lik_df["% Malignancy Indicator"]],
            textposition="top center",
            cliponaxis=False,
            line=dict(color="crimson", width=3)
        ))
        fig.update_layout(
            xaxis_title="Tumor Feature",
            yaxis_title="% Malignancy Indicator",
            yaxis_range=[0,100],
            height=500,
            margin=dict(l=10, r=10, t=10, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to show chart.")

    # Diagnosis Estimate
    st.subheader("Diagnosis Estimate")
    ordered = [f for grp in FEATURE_GROUPS.values() for f in grp]
    X = np.array([[values[f] for f in ordered]])
    p_mal = pipe.predict_proba(X)[0,1]

    if p_mal >= 0.5:
        label, prob, count, icon, fn = (
            "MALIGNANT", p_mal, int(round(p_mal*100)), "🚨", st.error
        )
    else:
        label, prob, count, icon, fn = (
            "BENIGN", 1-p_mal, int(round((1-p_mal)*100)), "✅", st.success
        )

    conf = "Low" if prob<0.6 else "Medium" if prob<0.85 else "High"

    fn(
        f"**{label}**  \n"
        f"Probability: **{prob:.1%}** ({conf} Confidence)  \n"
        f"{count} out of 100 similar cases were {label.lower()}.",
        icon=icon
    )

    st.caption("For educational use only; not medical advice.")
