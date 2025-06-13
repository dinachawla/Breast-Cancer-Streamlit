import streamlit as st, joblib, numpy as np
from pathlib import Path

MODEL_PATH = Path("breast_cancer_clf.pkl")
TEST_ACC   = 0.971

@st.cache_resource
def load_model(p): return joblib.load(p)

model = load_model(MODEL_PATH)

st.title("Breast-tumor classifier 🩺")
st.caption(f"Hold-out accuracy: {TEST_ACC:.1%}")

radius_mean  = st.number_input("Mean radius (mm)", 0.0, 50.0, step=.01)
texture_mean = st.number_input("Mean texture",     0.0,100.0, step=.01)
perimeter_mean = st.number_input("Mean perimeter", 0.0,300.0, step=.01)
area_mean    = st.number_input("Mean area",        0.0,2500.,step=1.)

if st.button("Classify"):
    X = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean]])
    p = model.predict_proba(X)[0,1]
    st.success(f"{'Malignant' if p>=.5 else 'Benign'}  (Prob: {p:.1%})")
