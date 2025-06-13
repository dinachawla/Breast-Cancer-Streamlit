import streamlit as st
import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path("breast_cancer_pipe.pkl")   #changed
TEST_ACC   = 0.971

@st.cache_resource
def load_model(path: Path):
    """Load the sklearn Pipeline once and keep it in memory."""
    return joblib.load(path)

pipe = load_model(MODEL_PATH)

st.title("Breast-tumor classifier 🩺")
st.caption(f"Hold-out accuracy: {TEST_ACC:.1%}")

st.subheader("Enter mean tumour metrics")
cols = st.columns(2)
radius_mean   = cols[0].number_input("Mean radius (mm)", 0.0, 50.0, step=.01)
texture_mean  = cols[1].number_input("Mean texture",     0.0,100.0, step=.01)
perimeter_mean= cols[0].number_input("Mean perimeter",   0.0,300.0, step=.01)
area_mean     = cols[1].number_input("Mean area (mm²)",  0.0,2500.,step=1.)

if st.button("Classify"):
    X = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean]])
    prob_malignant = pipe.predict_proba(X)[0, 1]
    label = "⚠️ **Malignant**" if prob_malignant >= 0.5 else "✅ **Benign**"
    st.markdown(f"{label} &nbsp; *(probability {prob_malignant:.1%})*")
    st.caption("Model is for educational use only, not medical advice.")
