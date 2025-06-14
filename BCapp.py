import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# ─── Settings ────────────────────────────────────────────────────────────────
MODEL_PATH = Path("breast_cancer_pipe.pkl")   # matches the file in your repo
TEST_ACC   = 0.971                            # hold-out accuracy you reported

# Hard-coded feature means (computed once from the dataset)
AVG = {
    "radius_mean":    14.127,   # mm
    "texture_mean":   19.289,
    "perimeter_mean": 91.969,   # mm
    "area_mean":     654.889    # mm²
}

# ─── Helper: load model once and cache ───────────────────────────────────────
@st.cache_resource
def load_model(path: Path):
    """Load the sklearn Pipeline once and keep it in memory."""
    return joblib.load(path)

pipe = load_model(MODEL_PATH)

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("Breast-tumor classifier 🩺")
st.caption(f"Hold-out accuracy: {TEST_ACC:.1%}")

st.subheader("Enter mean-value tumour metrics")

cols = st.columns(2)

radius_mean = cols[0].slider(
    "Mean radius (mm)",
    min_value=0.0,  max_value=50.0,
    value=AVG["radius_mean"],  step=0.01
)

texture_mean = cols[1].slider(
    "Mean texture",
    min_value=0.0,  max_value=100.0,
    value=AVG["texture_mean"], step=0.01
)

perimeter_mean = cols[0].slider(
    "Mean perimeter (mm)",
    min_value=0.0,  max_value=300.0,
    value=AVG["perimeter_mean"], step=0.01
)

area_mean = cols[1].slider(
    "Mean area (mm²)",
    min_value=0.0,  max_value=2500.0,
    value=AVG["area_mean"],    step=1.0
)

# ─── Prediction ──────────────────────────────────────────────────────────────
if st.button("Classify"):
    X = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean]])
    prob_malignant = pipe.predict_proba(X)[0, 1]
    label = "⚠️ **Malignant**" if prob_malignant >= 0.5 else "✅ **Benign**"
    st.markdown(f"{label} &nbsp; *(probability {prob_malignant:.1%})*")
    st.caption("Model is for educational use only, not medical advice.")
