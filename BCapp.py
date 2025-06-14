import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = Path("breast_cancer_pipe.pkl")
TEST_ACC   = 0.971

METRICS = {
    "radius_mean": {
        "label": "Mean radius (mm)",
        "desc":  "Average distance from the nucleus center to the cell border.",
        "min": 0.0,  "max": 50.0,   "step": 0.01,  "avg": 14.127,
    },
    "texture_mean": {
        "label": "Mean texture",
        "desc":  "Standard deviation of gray-scale values inside the nucleus—higher means more heterogeneity.",
        "min": 0.0,  "max": 100.0,  "step": 0.01,  "avg": 19.289,
    },
    "perimeter_mean": {
        "label": "Mean perimeter (mm)",
        "desc":  "Average length of the nucleus outline; relates to size and complexity.",
        "min": 0.0,  "max": 300.0,  "step": 0.01,  "avg": 91.969,
    },
    "area_mean": {
        "label": "Mean area (mm²)",
        "desc":  "Average two-dimensional area of the nucleus—larger areas indicate bigger nuclei.",
        "min": 0.0,  "max": 2500.0, "step": 1.0,   "avg": 654.889,
    },
}

@st.cache_resource
def load_model(p: Path):
    return joblib.load(p)

pipe = load_model(MODEL_PATH)

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("Breast-tumor classifier 🩺")
st.caption(f"Hold-out accuracy: {TEST_ACC:.1%}")
st.subheader("Enter mean-value tumour metrics")

values = {}
for key, cfg in METRICS.items():
    # Bigger, distinct metric heading
    st.markdown(f"<h4 style='margin-bottom:0.2rem'>{cfg['label']}</h4>",
                unsafe_allow_html=True)
    st.caption(cfg["desc"])
    values[key] = st.slider(
        label=" ", key=key,
        min_value=cfg["min"], max_value=cfg["max"],
        value=cfg["avg"], step=cfg["step"]
    )
    st.divider()          # neat horizontal line (Streamlit ≥1.25)

# ─── Prediction & explanation ────────────────────────────────────────────────
if st.button("Classify"):
    X = np.array([[values[k] for k in METRICS]])
    prob = pipe.predict_proba(X)[0, 1]        # P(malignant)

    if prob >= 0.5:
        st.markdown(f"⚠️ **Malignant** *(probability {prob:.1%})*")
        st.info(
            f"The model estimates a **{prob:.1%}** likelihood that the tumour "
            "is malignant. In 100 similar cases, about "
            f"**{prob*100:.0f}** would be malignant."
        )
    else:
        st.markdown(f"✅ **Benign** *(probability {1-prob:.1%})*")
        st.info(
            f"The model estimates a **{1-prob:.1%}** likelihood that the tumour "
            "is benign and **{prob:.1%}** malignant."
        )

    st.caption("Model is for educational use only; it does not replace professional medical advice.")
