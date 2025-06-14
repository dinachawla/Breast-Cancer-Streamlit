import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# ─── Settings ────────────────────────────────────────────────────────────────
MODEL_PATH = Path("breast_cancer_pipe.pkl")
TEST_ACC   = 0.971

AVG = {  # dataset-wide means
    "radius_mean":    14.127,
    "texture_mean":   19.289,
    "perimeter_mean": 91.969,
    "area_mean":     654.889
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
radius_mean = c1.slider("Mean radius (mm)",     0.0, 50.0,  AVG["radius_mean"],  step=0.01)
texture_mean = c2.slider("Mean texture",        0.0,100.0,  AVG["texture_mean"], step=0.01)
perimeter_mean = c1.slider("Mean perimeter (mm)",0.0,300.0, AVG["perimeter_mean"],step=0.01)
area_mean = c2.slider("Mean area (mm²)",        0.0,2500.0, AVG["area_mean"],    step=1.0)

# ─── Prediction & explanation ────────────────────────────────────────────────
if st.button("Classify"):
    X = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean]])
    prob = pipe.predict_proba(X)[0, 1]          # P(malignant)

    if prob >= 0.5:
        st.markdown(f"⚠️ **Malignant** *(probability {prob:.1%})*")
        st.info(
            "The model estimates a **{:.1%} likelihood** that the tumour is "
            "malignant.  That means, out of 100 similar cases, it would expect about "
            "{:.0f} to be malignant and the rest benign.  A higher probability "
            "indicates greater confidence in a malignant diagnosis.".format(prob, prob*100)
        )
    else:
        st.markdown(f"✅ **Benign** *(probability {1-prob:.1%})*")
        st.info(
            "The model estimates a **{:.1%} likelihood** that the tumour is benign "
            "and **{:.1%}** that it’s malignant.  Lower malignant probability "
            "induces greater confidence in a benign diagnosis, but it is **not a "
            "substitute for professional medical assessment**.".format(1-prob, prob)
        )

    st.caption("Model is for educational use only and should not inform clinical decisions.")
