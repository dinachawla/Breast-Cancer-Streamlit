# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:58:38.781661Z","iopub.execute_input":"2025-06-13T01:58:38.781907Z","iopub.status.idle":"2025-06-13T01:58:38.786632Z","shell.execute_reply.started":"2025-06-13T01:58:38.781888Z","shell.execute_reply":"2025-06-13T01:58:38.785901Z"}}
import sys, pathlib, os

repo = pathlib.Path("/kaggle/input/breast-cancer-classification-main")

# 1. Make its Python modules importable
sys.path.append(str(repo))

# 2. Install any Python dependencies the repo lists
req = repo / "requirements.txt"
if req.exists():
    !pip install -q -r {req}

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:58:38.787867Z","iopub.execute_input":"2025-06-13T01:58:38.788084Z","iopub.status.idle":"2025-06-13T01:58:38.941474Z","shell.execute_reply.started":"2025-06-13T01:58:38.788068Z","shell.execute_reply":"2025-06-13T01:58:38.940735Z"}}
# ── 1. standard imports ───────────────────────────────────────────────────────
from pathlib import Path          # pathlib is part of the standard library
import sys, os

# ── 2. point to the repo folder that Kaggle mounted ──────────────────────────
# Look in the right-hand “Data” panel → copy the exact folder name.
# For a public repo it’s usually <repo-name>-<branch>, all lower-case.
repo_root = Path("/kaggle/input/breast-cancer-classification")   # adjust if different

# ── 3. sanity-check: list the first few files ────────────────────────────────
!find $repo_root -maxdepth 2 -type f | head -n 20

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:58:38.942825Z","iopub.execute_input":"2025-06-13T01:58:38.943044Z","iopub.status.idle":"2025-06-13T01:58:39.075691Z","shell.execute_reply.started":"2025-06-13T01:58:38.943020Z","shell.execute_reply":"2025-06-13T01:58:39.074825Z"}}
!ls $repo_root

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:58:39.076623Z","iopub.execute_input":"2025-06-13T01:58:39.076851Z","iopub.status.idle":"2025-06-13T01:58:39.080871Z","shell.execute_reply.started":"2025-06-13T01:58:39.076833Z","shell.execute_reply":"2025-06-13T01:58:39.080155Z"}}
from pathlib import Path
import sys, runpy, inspect, subprocess

# **EDIT THIS** if the folder name in the right-hand Data pane is different
repo_root = Path("/kaggle/input/breast-cancer-classification")

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:58:39.082243Z","iopub.execute_input":"2025-06-13T01:58:39.082438Z","iopub.status.idle":"2025-06-13T01:58:40.682247Z","shell.execute_reply.started":"2025-06-13T01:58:39.082424Z","shell.execute_reply":"2025-06-13T01:58:40.681610Z"}}
# Writes bc_repo_script.py in the writable /kaggle/working/
subprocess.run([
    "jupyter", "nbconvert",
    "--to", "python",
    str(repo_root / "BC_Classification.ipynb"),
    "--output", "/kaggle/working/bc_repo_script"
], check=True)

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:58:40.683043Z","iopub.execute_input":"2025-06-13T01:58:40.683232Z","iopub.status.idle":"2025-06-13T01:58:42.301227Z","shell.execute_reply.started":"2025-06-13T01:58:40.683216Z","shell.execute_reply":"2025-06-13T01:58:42.300503Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:58:42.302271Z","iopub.execute_input":"2025-06-13T01:58:42.302445Z","iopub.status.idle":"2025-06-13T01:58:42.307358Z","shell.execute_reply.started":"2025-06-13T01:58:42.302433Z","shell.execute_reply":"2025-06-13T01:58:42.306755Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:58:42.308088Z","iopub.execute_input":"2025-06-13T01:58:42.308293Z","iopub.status.idle":"2025-06-13T01:58:42.323406Z","shell.execute_reply.started":"2025-06-13T01:58:42.308277Z","shell.execute_reply":"2025-06-13T01:58:42.322807Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:58:42.324118Z","iopub.execute_input":"2025-06-13T01:58:42.324351Z","iopub.status.idle":"2025-06-13T01:59:10.524869Z","shell.execute_reply.started":"2025-06-13T01:58:42.324333Z","shell.execute_reply":"2025-06-13T01:59:10.524032Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:59:10.525539Z","iopub.execute_input":"2025-06-13T01:59:10.525787Z","iopub.status.idle":"2025-06-13T01:59:13.120775Z","shell.execute_reply.started":"2025-06-13T01:59:10.525770Z","shell.execute_reply":"2025-06-13T01:59:13.120106Z"}}
# 🔧 Install Streamlit (and joblib) FOR THIS NOTEBOOK KERNEL
import sys, importlib.util, subprocess

subprocess.run([sys.executable, "-m", "pip", "install", "-qU", "streamlit", "joblib"])

# ✅ quick sanity-check
print("Streamlit found:", importlib.util.find_spec("streamlit") is not None)
print("Python path     :", sys.executable)

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:59:13.122586Z","iopub.execute_input":"2025-06-13T01:59:13.122788Z","iopub.status.idle":"2025-06-13T01:59:13.131033Z","shell.execute_reply.started":"2025-06-13T01:59:13.122776Z","shell.execute_reply":"2025-06-13T01:59:13.130387Z"}}
import pandas as pd
from pathlib import Path

# 1. Point to the real file inside the mounted dataset
DATA_PATH = Path("/kaggle/input/breast-cancer-classification")  # adjust folder name
CSV_FILE  = next(DATA_PATH.glob("*.csv"))                       # finds the first .csv
print("Reading:", CSV_FILE)

df = pd.read_csv(CSV_FILE)

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:59:13.131753Z","iopub.execute_input":"2025-06-13T01:59:13.132406Z","iopub.status.idle":"2025-06-13T01:59:13.193864Z","shell.execute_reply.started":"2025-06-13T01:59:13.132382Z","shell.execute_reply":"2025-06-13T01:59:13.193352Z"}}
raw_csv = (
    "https://raw.githubusercontent.com/"
    "dinachawla/Breast-Cancer-Classification/main/data.csv"
)

import pandas as pd
df = pd.read_csv(raw_csv)

print(df.shape)   # quick sanity-check
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:59:13.194498Z","iopub.execute_input":"2025-06-13T01:59:13.194722Z","iopub.status.idle":"2025-06-13T01:59:13.246613Z","shell.execute_reply.started":"2025-06-13T01:59:13.194704Z","shell.execute_reply":"2025-06-13T01:59:13.246002Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:59:13.247264Z","iopub.execute_input":"2025-06-13T01:59:13.247435Z","iopub.status.idle":"2025-06-13T01:59:13.252480Z","shell.execute_reply.started":"2025-06-13T01:59:13.247422Z","shell.execute_reply":"2025-06-13T01:59:13.251835Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T01:59:13.253227Z","iopub.execute_input":"2025-06-13T01:59:13.253427Z","iopub.status.idle":"2025-06-13T01:59:16.036683Z","shell.execute_reply.started":"2025-06-13T01:59:13.253411Z","shell.execute_reply":"2025-06-13T01:59:16.036038Z"}}
# ⬇️ Run this in a notebook code cell
!pip install --quiet streamlit

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