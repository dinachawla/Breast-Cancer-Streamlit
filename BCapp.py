import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# ─── Settings ────────────────────────────────────────────────────────────────
MODEL_PATH = Path("breast_cancer_pipe.pkl")
TEST_ACC   = 0.971

# Dataset-wide average values for each feature
AVG = {
    "radius_mean":    14.127,    # mm
    "texture_mean":   19.289,
    "perimeter_mean": 91.969,    # mm
    "area_mean":     654.889     # mm²
}

# ─── Helper ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

pipe = load_model(MODEL_PATH)

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("Breast-tumor classifier 🩺")
st.caption(f"Hold-out accuracy: {TEST_ACC:.1%}")

st.subheader("Enter mean-value tumour metrics")

c1, c2 = st.columns(2)

# Radius
c1.markdown(
    "##### Mean radius (mm)\n"
    "*Average distance from the nucleus center to the cell border. "
    "Larger radii generally indicate larger nuclei.*"
)
radius_mean = c1.slider(
    label=" ",  min_value=0.0, max_value=50.0,
    value=AVG["radius_mean"], step=0.01, key="radius"
)

# Texture
c2.markdown(
    "##### Mean texture\n"
    "*Standard deviation of gray-scale values inside the nucleus—"
    "a roughness measure. Higher texture means more heterogeneous tissue.*"
)
texture_mean = c2.slider(
    label=" ",  min_value=0.0, max_value=100.0,
    value=AVG["texture_mean"], step=0.01, key="texture"
)

# Perimeter
c1.markdown(
    "##### Mean perimeter (mm)\n"
    "*Average length of the nucleus outline. Related to size and shape complexity.*"
)
perimeter_mean = c1.slider(
    label=" ",  min_value=0.0, max_value=300.0,
    value=AVG["perimeter_mean"], step=0.01, key="perimeter"
)

# Area
c2.markdown(
    "##### Mean area (mm²)\n"
    "*Average two-dimensional area of the nucleus. Larger areas indicate bigger nuclei.*"
)
area_mean = c2.slider(
    label=" ",  min_value=0.0, max_value=2500.0,
    value=AVG["area_mean"], step=1.0, key="area"
)

# ─── Prediction & explanation ────────────────────────────────────────────────
if st.button("Classify"):
    X = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean]])
    prob = pipe.predict_proba(X)[0, 1]     # P(malignant)

    if prob >= 0.5:
        st.markdown(f"⚠️ **Malignant** *(probability {prob:.1%})*")
        st.info(
            "The model estimates a **{:.1%}** likelihood that the tumour is "
            "malignant. In 100 similar cases, it would expect roughly **{:.0f}** "
            "to be malignant.".format(prob, prob*100)
        )
    else:
        st.markdown(f"✅ **Benign** *(probability {1-prob:.1%})*")
        st.info(
            "The model estimates a **{:.1%}** likelihood that the tumour is benign "
            "and **{:.1%}** malignant. Lower malignant probability strengthens "
            "confidence in a benign outcome.".format(1-prob, prob)
        )

    st.caption("Model is for educational use only and should not inform clinical decisions.")
