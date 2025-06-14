import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# ─── Model & dataset settings ────────────────────────────────────────────────
MODEL_PATH = Path("breast_cancer_pipe.pkl")
TEST_ACC   = 0.971

# (key, label, description, min, max, slider_step, default/avg)
FEATURES = [
    ("radius_mean",    "Mean radius (mm)",
     "Average distance from the nucleus centre to the cell border.",
     0.0, 50.0, 0.01, 14.127),

    ("texture_mean",   "Mean texture",
     "Std-dev of gray-scale values inside the nucleus—higher means more heterogeneity.",
     0.0, 100.0, 0.01, 19.289),

    ("perimeter_mean", "Mean perimeter (mm)",
     "Average length of the nucleus outline; relates to size and complexity.",
     0.0, 300.0, 0.01, 91.969),

    ("area_mean",      "Mean area (mm²)",
     "Average two-dimensional area of the nucleus; larger areas indicate bigger nuclei.",
     0.0, 2500.0, 1.0, 654.889),
]

@st.cache_resource
def load_model(p: Path):
    return joblib.load(p)

pipe = load_model(MODEL_PATH)

# ─── Page header ─────────────────────────────────────────────────────────────
st.title("Breast Cancer ML Classifier 🩺")
st.markdown(
    "Estimate whether a breast-tumour sample is **benign** or **malignant**. "
    "Move each slider *or* type an exact value, then press **Classify**."
)
st.caption(f"Model hold-out accuracy: {TEST_ACC:.1%}")
st.subheader("Enter mean-value tumour metrics")

# ─── Custom CSS – accent colour + button style ───────────────────────────────
st.markdown(
    """
    <style>
      :root { --primary-color:#d7263d; }
      div.stButton > button:first-child{
        background-color:var(--primary-color);
        border-color:var(--primary-color);
        color:#fff;
        width:100%;
        font-size:1.1rem;
        font-weight:600;
        padding:0.6em 1.2em;
        border-radius:8px;
      }
      div.stButton > button:first-child:hover{ opacity:0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Grid of sliders + number inputs ─────────────────────────────────────────
values = {}
for row_start in range(0, len(FEATURES), 2):
    left, right = st.columns(2, gap="large")
    for col, cfg in zip((left, right), FEATURES[row_start : row_start + 2]):
        key, label, desc, vmin, vmax, step, avg = cfg
        with col:
            st.markdown(f"<h4 style='margin-bottom:0.2rem'>{label}</h4>",
                        unsafe_allow_html=True)
            st.caption(desc)
            st.markdown(f"*Population average: **{avg:.3f}***")

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

    if row_start + 2 < len(FEATURES):
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# Spacer before button
st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

# ─── Prediction & probability explanation ───────────────────────────────────
if st.button("Classify"):
    X = np.array([[values[k] for k, *_ in FEATURES]])
    p = pipe.predict_proba(X)[0, 1]        # P(malignant)

    if p >= 0.5:
        st.markdown(f"⚠️ **Malignant** *(probability {p:.1%})*")
        st.info(
            f"The model estimates a **{p:.1%}** chance the tumour is malignant. "
            f"Roughly **{p*100:.0f}** of 100 similar cases would be malignant."
        )
    else:
        st.markdown(f"✅ **Benign** *(probability {1-p:.1%})*")
        st.info(
            f"The model estimates a **{1-p:.1%}** chance the tumour is benign "
            f"and **{p:.1%}** malignant."
        )

    st.caption("Model is for educational use only and does not replace professional medical advice.")
