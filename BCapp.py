import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# ─── Model & dataset settings ────────────────────────────────────────────────
MODEL_PATH = Path("breast_cancer_pipe.pkl")
TEST_ACC   = 0.971   # hold-out accuracy

# Slider configuration: (key, label, description, min, max, step, default)
FEATURES = [
    ("radius_mean",    "Mean radius (mm)",
     "Average distance from the nucleus center to the cell border.",
     0.0, 50.0, 0.01, 14.127),

    ("texture_mean",   "Mean texture",
     "Standard deviation of gray-scale values inside the nucleus—higher means more heterogeneity.",
     0.0, 100.0, 0.01, 19.289),

    ("perimeter_mean", "Mean perimeter (mm)",
     "Average length of the nucleus outline; relates to size and complexity.",
     0.0, 300.0, 0.01, 91.969),

    ("area_mean",      "Mean area (mm²)",
     "Average two-dimensional area of the nucleus—larger areas indicate bigger nuclei.",
     0.0, 2500.0, 1.0, 654.889),
]

# ─── Helper to load the model once ───────────────────────────────────────────
@st.cache_resource
def load_model(p: Path):
    return joblib.load(p)

pipe = load_model(MODEL_PATH)

# ─── Page header ─────────────────────────────────────────────────────────────
st.title("Breast-tumor classifier 🩺")
st.caption(f"Hold-out accuracy: {TEST_ACC:.1%}")
st.subheader("Enter mean-value tumour metrics")

# ─── Grid of sliders (two columns, two rows) ─────────────────────────────────
values = {}
for row_start in range(0, len(FEATURES), 2):
    left, right = st.columns(2, gap="large")          # even 50/50 columns
    for col, cfg in zip((left, right), FEATURES[row_start:row_start + 2]):
        key, label, desc, minv, maxv, step, default = cfg
        with col:
            # Bold, larger heading for each metric
            st.markdown(f"<h4 style='margin-bottom:0.2rem'>{label}</h4>",
                        unsafe_allow_html=True)
            st.caption(desc)
            # An empty label keeps the slider tight under the caption
            values[key] = st.slider(
                label=" ", key=key,
                min_value=minv, max_value=maxv, value=default, step=step
            )

# ─── Prediction & probability explanation ───────────────────────────────────
if st.button("Classify"):
    X = np.array([[values[k] for k, *_ in FEATURES]])
    prob_malignant = pipe.predict_proba(X)[0, 1]

    if prob_malignant >= 0.5:
        st.markdown(f"⚠️ **Malignant** *(probability {prob_malignant:.1%})*")
        st.info(
            f"The model estimates a **{prob_malignant:.1%}** likelihood that the "
            "tumour is malignant. Out of 100 similar cases, about "
            f"**{prob_malignant*100:.0f}** would be malignant."
        )
    else:
        st.markdown(f"✅ **Benign** *(probability {1-prob_malignant:.1%})*")
        st.info(
            f"The model estimates a **{1-prob_malignant:.1%}** likelihood that the "
            "tumour is benign and **{prob_malignant:.1%}** malignant."
        )

    st.caption("Model is for educational use only; it is not a substitute for professional medical advice.")
