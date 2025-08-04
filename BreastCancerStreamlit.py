import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path
from sklearn.datasets import load_breast_cancer

st.set_page_config(layout="wide")

# sticky CSS
st.markdown("""
    <style>
    .sticky-right {
        position: -webkit-sticky;
        position: sticky;
        top: 1rem;
        align-self: start;
        background-color: white;
        padding-top: 0.5rem;
        z-index: 10;
    }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("breast_cancer_pipe_11features.pkl")
TEST_ACC = 0.965

# load data & model
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = 1 - data.target    # 1=malignant
pipe = joblib.load(MODEL_PATH)

# slider bounds
SELECTED_FEATURES = [
    "mean radius","worst radius","mean perimeter","worst perimeter",
    "mean area","worst area","mean concavity","worst concavity",
    "mean concave points","worst concave points","mean texture"
]
percentile_bounds = {
    col: (
      np.percentile(df[col],5),
      np.percentile(df[col],95),
      df[col].mean(),
      0.001 if np.ptp(df[col])<10 else 0.1 if np.ptp(df[col])<100 else 1
    )
    for col in SELECTED_FEATURES
}

FEATURE_GROUPS = {
    "Radius (mm)": ["mean radius","worst radius"],
    "Perimeter (mm)": ["mean perimeter","worst perimeter"],
    "Area (mm²)": ["mean area","worst area"],
    "Concavity": ["mean concavity","worst concavity"],
    "Concave Points": ["mean concave points","worst concave points"],
    "Texture": ["mean texture"]
}

# two tabs
tab_manual, tab_auto = st.tabs(["🔧 Manual", "🤖 Automatic"])

### Manual mode (your existing sliders + chart + diagnosis)
with tab_manual:
    st.title("Manual Mode")
    st.caption(f"Model hold-out accuracy: {TEST_ACC:.1%}")
    left, right = st.columns([1,1], gap="large")

    # keep session state for sync/reset
    if "vals" not in st.session_state:
        st.session_state.vals = {k:percentile_bounds[k][2] for k in SELECTED_FEATURES}

    def _sync(k):
      st.session_state.vals[k] = st.session_state[f"s_{k}"]

    with left:
        for grp, keys in FEATURE_GROUPS.items():
            st.markdown(f"### {grp}")
            cols = st.columns(len(keys))
            for col,key in zip(cols,keys):
                low,high,avg,step = percentile_bounds[key]
                with col:
                    st.markdown(f"**{key.title()}**  \n_avg: {avg:.3f}_")
                    s = st.slider("", key=f"s_{key}",
                                 min_value=float(low), max_value=float(high),
                                 step=float(step),
                                 value=st.session_state.vals[key],
                                 on_change=_sync, args=(key,))
                    st.session_state.vals[key] = s

    with right:
        st.markdown("<div class='sticky-right'>", unsafe_allow_html=True)
        st.subheader("Feature-Level Malignancy Likelihood")
        lik = []
        for f,v in st.session_state.vals.items():
            m = df[(df[f] >= (v*0.95)) & (df[f] <= (v*1.05))]['target'].mean()
            lik.append((f, v, 100*m))
        lik_df = pd.DataFrame(lik, columns=["Feature","Value","% Malignant"]).dropna()
        fig = go.Figure([go.Scatter(
            x=lik_df["Feature"], y=lik_df["% Malignant"],
            mode="lines+markers+text",
            text=[f"{x:.1f}%" for x in lik_df["% Malignant"]],
            textposition="top center", line=dict(width=3)
        )])
        fig.update_layout(yaxis=dict(range=[0,100]), height=400,
                          margin=dict(l=10,r=10,t=10,b=40))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Diagnosis Estimate")
        X = pd.DataFrame([st.session_state.vals])[pipe.feature_names_in_]
        p = pipe.predict_proba(X)[0,1]
        if p>=0.5:
            st.error(f"🚨 MALIGNANT {p:.1%}", icon="🚨")
        else:
            st.success(f"✅ BENIGN {(1-p):.1%}", icon="✅")
        st.markdown("</div>", unsafe_allow_html=True)


### Automatic mode (one slider → back-calc feature values)
with tab_auto:
    st.title("Automatic Mode")
    st.caption("Pick a malignancy probability and see what feature values you’d need (others held at their population mean).")

    # pull scaler & LR
    scaler = pipe.named_steps["standardscaler"]
    lr     = pipe.named_steps["logisticregression"]
    coefs  = lr.coef_[0]
    intercept = lr.intercept_[0]
    means = scaler.mean_
    scales = scaler.scale_

    # desired p
    p_des = st.slider("Desired malignancy probability", 0.0, 1.0, 0.5, 0.01)
    logit = np.log(p_des/(1-p_des))

    req = {}
    for i, feat in enumerate(pipe.feature_names_in_):
        w = coefs[i]
        if abs(w)<1e-6:
            req[feat] = np.nan
        else:
            # solve logit = intercept + Σ w_j * xj_scaled
            # assume xj_scaled=0 for all j≠i → xi_scaled = (logit-intercept)/w
            xi_s = (logit - intercept)/w
            # back to original units
            req[feat] = xi_s * scales[i] + means[i]

    req_df = pd.DataFrame.from_dict(req, orient="index", columns=["Required value"])
    req_df.index.name = "Feature"
    st.dataframe(req_df.loc[SELECTED_FEATURES].round(3))
