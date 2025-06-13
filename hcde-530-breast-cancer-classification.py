# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:19:05.494820Z","iopub.execute_input":"2025-06-13T01:19:05.495121Z","iopub.status.idle":"2025-06-13T01:19:05.501872Z","shell.execute_reply.started":"2025-06-13T01:19:05.495095Z","shell.execute_reply":"2025-06-13T01:19:05.500639Z"}}
import sys, pathlib, os

repo = pathlib.Path("/kaggle/input/breast-cancer-classification-main")

# 1. Make its Python modules importable
sys.path.append(str(repo))

# 2. Install any Python dependencies the repo lists
req = repo / "requirements.txt"
if req.exists():
    !pip install -q -r {req}

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:19:05.503334Z","iopub.execute_input":"2025-06-13T01:19:05.503623Z","iopub.status.idle":"2025-06-13T01:19:05.663202Z","shell.execute_reply.started":"2025-06-13T01:19:05.503598Z","shell.execute_reply":"2025-06-13T01:19:05.661835Z"}}
# ── 1. standard imports ───────────────────────────────────────────────────────
from pathlib import Path          # pathlib is part of the standard library
import sys, os

# ── 2. point to the repo folder that Kaggle mounted ──────────────────────────
# Look in the right-hand “Data” panel → copy the exact folder name.
# For a public repo it’s usually <repo-name>-<branch>, all lower-case.
repo_root = Path("/kaggle/input/breast-cancer-classification")   # adjust if different

# ── 3. sanity-check: list the first few files ────────────────────────────────
!find $repo_root -maxdepth 2 -type f | head -n 20

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:19:05.664320Z","iopub.execute_input":"2025-06-13T01:19:05.664641Z","iopub.status.idle":"2025-06-13T01:19:05.788888Z","shell.execute_reply.started":"2025-06-13T01:19:05.664608Z","shell.execute_reply":"2025-06-13T01:19:05.787858Z"}}
!ls $repo_root

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:19:05.791684Z","iopub.execute_input":"2025-06-13T01:19:05.792123Z","iopub.status.idle":"2025-06-13T01:19:05.797753Z","shell.execute_reply.started":"2025-06-13T01:19:05.792089Z","shell.execute_reply":"2025-06-13T01:19:05.796921Z"}}
from pathlib import Path
import sys, runpy, inspect, subprocess

# **EDIT THIS** if the folder name in the right-hand Data pane is different
repo_root = Path("/kaggle/input/breast-cancer-classification")

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:19:05.798577Z","iopub.execute_input":"2025-06-13T01:19:05.798833Z","iopub.status.idle":"2025-06-13T01:19:08.711551Z","shell.execute_reply.started":"2025-06-13T01:19:05.798805Z","shell.execute_reply":"2025-06-13T01:19:08.710627Z"}}
# Writes bc_repo_script.py in the writable /kaggle/working/
subprocess.run([
    "jupyter", "nbconvert",
    "--to", "python",
    str(repo_root / "BC_Classification.ipynb"),
    "--output", "/kaggle/working/bc_repo_script"
], check=True)

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:19:08.712585Z","iopub.execute_input":"2025-06-13T01:19:08.712959Z","iopub.status.idle":"2025-06-13T01:19:10.994104Z","shell.execute_reply.started":"2025-06-13T01:19:08.712925Z","shell.execute_reply":"2025-06-13T01:19:10.993117Z"}}
# Convert as before …
subprocess.run(["jupyter", "nbconvert", "--to", "python",
                str(repo_root / "BC_Classification.ipynb"),
                "--output", "/kaggle/working/bc_repo_script"], check=True)

# --- read the .py file, patch the bad line, write it back -------------------
import re, pathlib
script_file = pathlib.Path("/kaggle/working/bc_repo_script.py")
src = script_file.read_text()

patched = re.sub(
    r"\(x_data-np\.min\(x_data\)\)/\(np\.max\(x_data\)-np\.min\(x_data\)\)\.values",
    "(x_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0))",
    src, count=1)

script_file.write_text(patched)

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:19:10.995113Z","iopub.execute_input":"2025-06-13T01:19:10.995484Z","iopub.status.idle":"2025-06-13T01:19:11.003204Z","shell.execute_reply.started":"2025-06-13T01:19:10.995449Z","shell.execute_reply":"2025-06-13T01:19:11.002160Z"}}
import pathlib, re

# 1️⃣  Path to the converted .py script produced by nbconvert
script_file = pathlib.Path("/kaggle/working/bc_repo_script.py")

# 2️⃣  Read the file, insert "df = data" immediately after the read_csv line
src = script_file.read_text()
patched = re.sub(
    r"data\s*=\s*pd\.read_csv\('data\.csv'\)",
    r"data = pd.read_csv('data.csv')\n\ndf = data  # alias so later cells work",
    src,
    count=1,
)
script_file.write_text(patched)
print("✔ Added 'df = data' alias inside bc_repo_script.py")

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:19:11.004413Z","iopub.execute_input":"2025-06-13T01:19:11.004731Z","iopub.status.idle":"2025-06-13T01:19:11.038841Z","shell.execute_reply.started":"2025-06-13T01:19:11.004703Z","shell.execute_reply":"2025-06-13T01:19:11.037889Z"}}
# 🔧 1-time patch: replace MSE[optimal_k] with the correct index lookup
import pathlib, re

script_path = pathlib.Path("/kaggle/working/bc_repo_script.py")   # nbconvert output
src = script_path.read_text()

# Change only the first occurrence (that print line)
patched = re.sub(
    r"MSE\[\s*optimal_k\s*\]",
    "MSE[neighbours.index(optimal_k)]",
    src,
    count=1
)

script_path.write_text(patched)
print("✔ Patched bc_repo_script.py – IndexError fixed")

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:19:11.039818Z","iopub.execute_input":"2025-06-13T01:19:11.040147Z","iopub.status.idle":"2025-06-13T01:19:58.517936Z","shell.execute_reply.started":"2025-06-13T01:19:11.040116Z","shell.execute_reply":"2025-06-13T01:19:58.516951Z"}}
# ── Execute the converted script with the right working directory ──
import os, runpy, pandas as pd, numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator

repo_root = Path("/kaggle/input/breast-cancer-classification")   # edit if different
script_path = "/kaggle/working/bc_repo_script.py"                     # produced by nbconvert

cwd = os.getcwd()                 # remember where we started
os.chdir(repo_root)               # now 'data.csv' is in "."

try:
    ns = runpy.run_path(script_path)   # run the notebook code ➜ namespace dict
finally:
    os.chdir(cwd)                # ALWAYS go back, even if an error occurs

# ── Quick inspection of what the notebook created ───────────────────────────
model_candidates = {k: v for k, v in ns.items() if isinstance(v, BaseEstimator)}
print("🟢  Candidate model variables:")
for name, est in model_candidates.items():
    print(f"   • {name:25s} → {est.__class__.__name__}")

arrays = {k: v for k, v in ns.items()
          if isinstance(v, (pd.DataFrame, np.ndarray, list))}
first_len = len(next(iter(arrays.values()))) if arrays else None
print("\n🟢  Arrays/DataFrames with that same length (likely X_test / y_test):")
for name, arr in arrays.items():
    if len(arr) == first_len:
        print(f"   • {name:25s} → shape/len = {getattr(arr, 'shape', len(arr))}")

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:19:58.520933Z","iopub.execute_input":"2025-06-13T01:19:58.521394Z","iopub.status.idle":"2025-06-13T01:20:02.989901Z","shell.execute_reply.started":"2025-06-13T01:19:58.521371Z","shell.execute_reply":"2025-06-13T01:20:02.988958Z"}}
# 🔧 Install Streamlit (and joblib) FOR THIS NOTEBOOK KERNEL
import sys, importlib.util, subprocess

subprocess.run([sys.executable, "-m", "pip", "install", "-qU", "streamlit", "joblib"])

# ✅ quick sanity-check
print("Streamlit found:", importlib.util.find_spec("streamlit") is not None)
print("Python path     :", sys.executable)

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:20:02.991061Z","iopub.execute_input":"2025-06-13T01:20:02.991723Z","iopub.status.idle":"2025-06-13T01:20:03.009551Z","shell.execute_reply.started":"2025-06-13T01:20:02.991690Z","shell.execute_reply":"2025-06-13T01:20:03.008613Z"}}
import pandas as pd
from pathlib import Path

# 1. Point to the real file inside the mounted dataset
DATA_PATH = Path("/kaggle/input/breast-cancer-classification")  # adjust folder name
CSV_FILE  = next(DATA_PATH.glob("*.csv"))                       # finds the first .csv
print("Reading:", CSV_FILE)

df = pd.read_csv(CSV_FILE)

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:20:03.010689Z","iopub.execute_input":"2025-06-13T01:20:03.011016Z","iopub.status.idle":"2025-06-13T01:20:03.379587Z","shell.execute_reply.started":"2025-06-13T01:20:03.010986Z","shell.execute_reply":"2025-06-13T01:20:03.378674Z"}}
raw_csv = (
    "https://raw.githubusercontent.com/"
    "dinachawla/Breast-Cancer-Classification/main/data.csv"
)

import pandas as pd
df = pd.read_csv(raw_csv)

print(df.shape)   # quick sanity-check
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:20:03.380610Z","iopub.execute_input":"2025-06-13T01:20:03.380951Z","iopub.status.idle":"2025-06-13T01:20:03.900758Z","shell.execute_reply.started":"2025-06-13T01:20:03.380920Z","shell.execute_reply":"2025-06-13T01:20:03.899875Z"}}
# train_model.ipynb  – run this once
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# ------------------------------------------------------------------
# 1 · Load data
# ------------------------------------------------------------------
CSV_URL = (
    "https://raw.githubusercontent.com/"
    "dinachawla/Breast-Cancer-Classification/main/data.csv"
)
df = pd.read_csv(CSV_URL)

# Optional: drop a non-numeric ID column if it exists
if "id" in df.columns:
    df = df.drop(columns=["id"])

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# ------------------------------------------------------------------
# 2 · Train/validation split
# ------------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 3 · Build preprocessing + model pipeline
# ------------------------------------------------------------------
pipe = make_pipeline(
    SimpleImputer(strategy="median"),
    MinMaxScaler(),
    KNeighborsClassifier(n_neighbors=7)
)

pipe.fit(X_tr, y_tr)
print(f"Test accuracy: {pipe.score(X_te, y_te):.3f}")

# ------------------------------------------------------------------
# 4 · Persist artefact
# ------------------------------------------------------------------
joblib.dump(pipe, "knn_pipe.pkl")
print("Saved pipeline ➜ knn_pipe.pkl")

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:20:03.901946Z","iopub.execute_input":"2025-06-13T01:20:03.903065Z","iopub.status.idle":"2025-06-13T01:20:03.909566Z","shell.execute_reply.started":"2025-06-13T01:20:03.903033Z","shell.execute_reply":"2025-06-13T01:20:03.908547Z"}}
%%writefile breast_cancer_knn_app.py
import streamlit as st
import joblib, numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Resolve the model file regardless of environment
# ─────────────────────────────────────────────────────────────
if "__file__" in globals():
    MODEL_PATH = Path(__file__).with_name("knn_pipe.pkl")
else:                                   # running inside a notebook
    MODEL_PATH = Path("knn_pipe.pkl").resolve()

@st.cache_resource
def load_pipe():
    return joblib.load(MODEL_PATH)

pipe = load_pipe()

FEATURES = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean",
    "fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se",
    "smoothness_se","compactness_se","concavity_se","concave points_se",
    "symmetry_se","fractal_dimension_se","radius_worst","texture_worst",
    "perimeter_worst","area_worst","smoothness_worst","compactness_worst",
    "concavity_worst","concave points_worst","symmetry_worst",
    "fractal_dimension_worst",
]

st.set_page_config("Breast-Cancer K-N N Demo", layout="wide")
st.title("🩺 Breast-Cancer Diagnosis — K-Nearest Neighbours")

cols = st.columns(3)
user_vals = [
    st.number_input(f, min_value=0.0, value=0.0, step=0.001, format="%.4f",
                    key=f)               # key avoids Streamlit duplicates
    for f in FEATURES
]

if st.button("Predict"):
    proba = pipe.predict_proba([user_vals])[0, 1]
    label = "Malignant" if proba >= 0.5 else "Benign"

    st.metric("Prediction", label)
    st.progress(proba)
    st.write(f"Probability of malignancy: **{proba*100:.1f}%**")

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:20:03.910519Z","iopub.execute_input":"2025-06-13T01:20:03.910770Z","iopub.status.idle":"2025-06-13T01:20:05.038209Z","shell.execute_reply.started":"2025-06-13T01:20:03.910750Z","shell.execute_reply":"2025-06-13T01:20:05.037140Z"}}
import streamlit as st
import joblib, numpy as np
from pathlib import Path

# ───────────────────────────────────────────────────────────────────────────
# Resolve the model file regardless of environment
# ───────────────────────────────────────────────────────────────────────────
if "__file__" in globals():
    MODEL_PATH = Path(__file__).with_name("knn_pipe.pkl")
else:                                  # e.g. running in a notebook cell
    MODEL_PATH = Path("knn_pipe.pkl").resolve()

@st.cache_resource
def load_pipe():
    return joblib.load(MODEL_PATH)

pipe = load_pipe()

FEATURES = [                     # same order as training
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave points_worst", "symmetry_worst",
    "fractal_dimension_worst",
]

st.set_page_config("Breast-Cancer K-NN Demo", layout="wide")
st.title("🩺 Breast-Cancer Diagnosis — K-Nearest Neighbours")

cols = st.columns(3)
user_vals = []
for i, feat in enumerate(FEATURES):
    with cols[i % 3]:
        val = st.number_input(feat, min_value=0.0, value=0.0, step=0.001, format="%.4f")
        user_vals.append(val)

if st.button("Predict"):
    proba = pipe.predict_proba([user_vals])[0, 1]
    label = "Malignant" if proba >= 0.5 else "Benign"

    st.metric("Prediction", label)
    st.progress(proba)
    st.write(f"Probability of malignancy: **{proba*100:.1f}%**")

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:20:05.039124Z","iopub.execute_input":"2025-06-13T01:20:05.039363Z","iopub.status.idle":"2025-06-13T01:20:05.204353Z","shell.execute_reply.started":"2025-06-13T01:20:05.039344Z","shell.execute_reply":"2025-06-13T01:20:05.203265Z"}}
!ps aux | grep streamlit | grep -v grep

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:20:05.205720Z","iopub.execute_input":"2025-06-13T01:20:05.206073Z","iopub.status.idle":"2025-06-13T01:20:08.901913Z","shell.execute_reply.started":"2025-06-13T01:20:05.206032Z","shell.execute_reply":"2025-06-13T01:20:08.900842Z"}}
%%bash
# 1. (re-)install streamlit if you need it
pip install -q streamlit

# 2. fire off Streamlit into the background and detach it
nohup streamlit run /kaggle/working/breast_cancer_knn_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true \
  > /kaggle/working/streamlit.log 2>&1 &

echo "▶️ Streamlit should now be running (logs → /kaggle/working/streamlit.log)"

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:20:08.902993Z","iopub.execute_input":"2025-06-13T01:20:08.903413Z","iopub.status.idle":"2025-06-13T01:20:13.924248Z","shell.execute_reply.started":"2025-06-13T01:20:08.903379Z","shell.execute_reply":"2025-06-13T01:20:13.923374Z"}}
import subprocess, time, IPython.display as ipd

# Kill anything already on the port
subprocess.run("fuser -k 8501/tcp || true", shell=True)

# Start Streamlit in the background
proc = subprocess.Popen(
    ["streamlit", "run", "breast_cancer_knn_app.py",
     "--server.address", "0.0.0.0",
     "--server.port", "8501",
     "--server.headless", "true"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)

# Wait a few seconds, then show it
time.sleep(5)
app_path = "/proxy/8501/"
ipd.display(ipd.HTML(f'<a target="_blank" href="{app_path}">🔗 Open the Streamlit app</a>'))
ipd.display(ipd.IFrame(src=app_path, width="100%", height=700))

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:20:13.925181Z","iopub.execute_input":"2025-06-13T01:20:13.925656Z","iopub.status.idle":"2025-06-13T01:20:14.060548Z","shell.execute_reply.started":"2025-06-13T01:20:13.925626Z","shell.execute_reply":"2025-06-13T01:20:14.059374Z"}}
!ls -lh /kaggle/working

# %% [code]
