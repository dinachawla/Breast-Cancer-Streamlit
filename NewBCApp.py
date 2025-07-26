import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path
from sklearn.datasets import load_breast_cancer

st.set_page_config(layout="wide")

# Inject CSS for scroll behavior and sticky chart
st.markdown("""
<style>
.scrollable-metrics {
    max-height: 80vh;
    overflow-y: auto;
    padding-right: 1rem;
}
.sticky-chart {
    position: sticky;
    top: 0;
    z-index: 999;
    background-color: white;
    padding-bottom: 1rem;
    border-bottom: 1px solid #ccc;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("breast_cancer_pipe_updated.pkl")
TEST_ACC = 0.965

# Load dataset and model
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

pipe = load_model(MODEL_PATH)

# Compute percentiles
percentile_bounds = {}
for col in data.feature_names:
    low = np.percentile(df[col], 5)
    high = np.percentile(df[col], 95)
    avg = df[col].mean()
    step = 0.001 if high - low < 10 else 0.1 if high - low < 100 else 1
    percentile_bounds[col] = (low, high, step, avg)

# Define grouped features
FEATURE_GROUPS = {
    "Radius (mm)": ["mean radius", "worst radius"],
    "Perimeter (mm)": ["mean perimeter", "worst perimeter"],
    "Area (mm²)": ["mean area", "worst area"],
    "Concavity": ["mean concavity", "worst concavity"],
    "Concave Points": ["mean concave points", "worst concave points"],
    "Texture": ["mean texture"]
}

# Init session state
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = None

for key in data.feature_names:
    if f"s_{key}" not in st.session_state or f"n_{key}" not in st.session_state:
        _, _, _, avg = percentile_bounds[key]
        st.session_state[f"s_{key}"] = avg
        st.session_state[f"n_{key}"] = avg

# Handle reset (safely)
if st.session_state.reset_trigger is not None:
    reset_key = st.session_state.reset_trigger
    if reset_key in percentile_bounds:
        _, _, _, avg = percentile_bounds[reset_key]
        st.session_state[f"s_{reset_key}"] = avg
        st.session_state[f"n_{reset_key}"] = avg
    st.session_state.reset_trigger = None

# Sync logic
def sync_slider(key):
    st.session_state[f"s_{key}"] = st.session_state[f"n_{key}"]

def sync_number_input(key):
    st.session_state[f"n_{key}"] = st.session_state[f"s_{key}"]

# UI
st.title("Breast Cancer ML Classifier 
